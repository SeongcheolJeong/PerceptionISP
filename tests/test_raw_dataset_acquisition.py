from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.raw_dataset_acquisition import (
    build_raw_dataset_acquisition,
    main as raw_dataset_acquisition_main,
    write_raw_dataset_acquisition,
)


class RawDatasetAcquisitionTest(unittest.TestCase):
    def test_offline_manifest_prioritizes_annotation_before_large_raw(self) -> None:
        summary = build_raw_dataset_acquisition(check_network=False)

        self.assertEqual(summary["status"], "pass")
        self.assertFalse(summary["network_checked"])
        self.assertEqual(summary["recommended_first"], "AODRaw:annotations_google_drive")
        resources = {(row["dataset"], row["resource"]): row for row in summary["resources"]}
        self.assertEqual(resources[("AODRaw", "annotations_google_drive")]["acquisition_status"], "manual_access_likely")
        self.assertEqual(resources[("AODRaw", "images_downsampled_raw_test_baidu")]["acquisition_status"], "manual_access_likely")
        self.assertEqual(resources[("AODRaw", "images_downsampled_raw_train_baidu")]["acquisition_status"], "defer_large_download")
        self.assertEqual(resources[("LOD", "raw_adapter_lod_google_drive")]["acquisition_status"], "manual_access_likely")
        self.assertIn("LOD_BMVC21", resources[("LOD", "raw_adapter_lod_google_drive")]["first_action"])
        self.assertIn("pwd=2021", resources[("LOD", "raw_dark_images_baidu")]["url"])
        self.assertIn("137.19", resources[("AODRaw", "images_downsampled_raw_train_baidu")]["blocker"])
        datasets = {row["dataset"]: row for row in summary["datasets"]}
        self.assertGreaterEqual(datasets["AODRaw"]["manual_access_count"], 1)
        self.assertGreaterEqual(datasets["AODRaw"]["large_download_count"], 1)

    def test_network_check_uses_opener_and_marks_failures(self) -> None:
        def opener(url: str, timeout: float) -> dict:
            if "drive.google.com" in url:
                return {"checked": True, "status": "error", "error": "unit failure"}
            return {"checked": True, "status": "reachable", "http_status": 200, "final_url": url}

        summary = build_raw_dataset_acquisition(check_network=True, opener=opener)

        self.assertTrue(summary["network_checked"])
        resources = {(row["dataset"], row["resource"]): row for row in summary["resources"]}
        failed = resources[("AODRaw", "annotations_google_drive")]
        self.assertEqual(failed["acquisition_status"], "blocked_by_network")
        self.assertEqual(failed["blocker"], "unit failure")
        self.assertEqual(summary["recommended_first"], "ROD / RAOD:openi_dataset")

    def test_local_state_recommends_aodraw_test_raw_after_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets"
            download_root = root / "downloads"
            (dataset_root / "aodraw" / "annotations").mkdir(parents=True)
            (dataset_root / "sid" / "downloads").mkdir(parents=True)
            (dataset_root / "lod" / "downloads").mkdir(parents=True)
            download_root.mkdir()
            (dataset_root / "aodraw" / "annotations" / "AODRaw_annotations.zip").write_bytes(b"annotations")
            (dataset_root / "sid" / "downloads" / "Sony2025.zip").write_bytes(b"partial-sid")
            with (dataset_root / "lod" / "downloads" / "LOD_BMVC2021.zip").open("wb") as handle:
                handle.truncate(22_000_000_000)
            (download_root / "AODRaw_downsampled_srgb.zip").write_bytes(b"partial")

            summary = build_raw_dataset_acquisition(
                check_network=False,
                check_local_state=True,
                dataset_root=dataset_root,
                download_roots=(download_root,),
            )

            self.assertTrue(summary["local_state_checked"])
            self.assertEqual(summary["recommended_first"], "AODRaw:images_downsampled_raw_test_baidu")
            self.assertTrue(summary["local_state"]["aodraw_annotations_present"])
            self.assertEqual(summary["local_state"]["aodraw_test_raw_zip"]["status"], "missing")
            self.assertEqual(summary["local_state"]["aodraw_srgb_zip"]["status"], "partial")
            self.assertEqual(summary["local_state"]["sid_sony_zip"]["status"], "partial")
            self.assertEqual(summary["local_state"]["lod_raw_adapter_archive"]["status"], "present")
            resources = {(row["dataset"], row["resource"]): row for row in summary["resources"]}
            self.assertEqual(resources[("AODRaw", "annotations_google_drive")]["acquisition_status"], "local_available")
            self.assertEqual(resources[("AODRaw", "images_downsampled_srgb_baidu")]["acquisition_status"], "local_invalid_retry")
            self.assertEqual(resources[("LOD", "raw_adapter_lod_google_drive")]["acquisition_status"], "local_available")
            self.assertEqual(resources[("SID", "sony_raw_google_storage")]["acquisition_status"], "local_invalid_retry")
            self.assertIn("local file is only", resources[("AODRaw", "images_downsampled_srgb_baidu")]["blocker"])

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_raw_dataset_acquisition(check_network=False)
            html_path = write_raw_dataset_acquisition(summary, root / "report")
            self.assertTrue((html_path.parent / "raw_dataset_acquisition_summary.json").exists())
            html = html_path.read_text()
            self.assertIn("PerceptionISP RAW Dataset Acquisition Audit", html)
            self.assertIn("AODRaw", html)
            self.assertIn("ROD / RAOD", html)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = raw_dataset_acquisition_main(["--output-dir", str(root / "cli"), "--check-local-state", "--dataset-root", str(root / "datasets")])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertFalse(printed["network_checked"])
            self.assertTrue(printed["local_state_checked"])
            self.assertEqual(printed["recommended_first"], "AODRaw:annotations_google_drive")
            self.assertTrue((root / "cli" / "raw_dataset_acquisition_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
