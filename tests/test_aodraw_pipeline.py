from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.aodraw_pipeline import (
    main as aodraw_pipeline_main,
    run_aodraw_pipeline,
    write_aodraw_pipeline,
)


class AODRawPipelineTest(unittest.TestCase):
    def test_pipeline_waits_when_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest()))

            summary = run_aodraw_pipeline(
                manifest=manifest,
                download_roots=(root / "Downloads",),
                dataset_root=root / "aodraw",
                output_dir=root / "pipeline",
                skip_eval=False,
                no_visuals=True,
                rgb_detector="numpy",
                count=1,
                width=8,
                height=8,
            )

            self.assertEqual(summary["status"], "waiting_for_files")
            self.assertFalse(summary["evaluation_ready"])
            self.assertEqual(summary["evaluation_status"], "blocked_by_availability")
            self.assertEqual(summary["availability"]["missing_file_count"], 2)

    def test_pipeline_imports_and_runs_eval_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "Downloads"
            _write_sample_files(downloads)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest()))

            summary = run_aodraw_pipeline(
                manifest=manifest,
                download_roots=(downloads,),
                dataset_root=root / "aodraw",
                output_dir=root / "pipeline",
                skip_eval=False,
                no_visuals=True,
                rgb_detector="numpy",
                count=1,
                width=8,
                height=8,
            )

            self.assertEqual(summary["status"], "evaluation_pass")
            self.assertTrue(summary["evaluation_ready"])
            self.assertEqual(summary["evaluation_status"], "pass")
            self.assertTrue((root / "aodraw" / "images_downsampled_raw" / "00000001.npy").exists())
            self.assertTrue((root / "pipeline" / "evaluation" / "comparison_summary.json").exists())

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest()))
            summary = run_aodraw_pipeline(
                manifest=manifest,
                download_roots=(root / "Downloads",),
                dataset_root=root / "aodraw",
                output_dir=root / "pipeline",
                dry_run=True,
                no_visuals=True,
                rgb_detector="numpy",
                count=1,
                width=8,
                height=8,
            )
            html_path = write_aodraw_pipeline(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_pipeline_summary.json").exists())
            self.assertIn("AODRaw Post-download Pipeline", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aodraw_pipeline_main(
                    [
                        "--manifest",
                        str(manifest),
                        "--download-root",
                        str(root / "Downloads"),
                        "--dataset-root",
                        str(root / "aodraw"),
                        "--output-dir",
                        str(root / "cli"),
                        "--dry-run",
                        "--rgb-detector",
                        "numpy",
                        "--count",
                        "1",
                        "--width",
                        "8",
                        "--height",
                        "8",
                        "--no-visuals",
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "dry_run")
            self.assertTrue((root / "cli" / "aodraw_pipeline_summary.json").exists())


def _manifest() -> list[dict]:
    return [
        {
            "image_id": 1,
            "file_name": "00000001.JPG",
            "selection_condition": "low_light",
            "tags": ["low_light"],
            "width": 8,
            "height": 8,
            "box_count": 1,
            "expected_raw_relative_path": "images_downsampled_raw/00000001.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000001.JPG",
            "boxes": [{"xyxy": [2.0, 2.0, 6.0, 6.0], "label": "person", "area": 16.0}],
        }
    ]


def _write_sample_files(root: Path) -> None:
    raw_path = root / "images_downsampled_raw" / "00000001.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic = np.zeros((8, 8), dtype=np.uint16)
    mosaic[2:6, 2:6] = 900
    np.save(raw_path, mosaic)
    srgb_path = root / "images_downsampled_srgb" / "00000001.JPG"
    srgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[2:6, 2:6, :] = 180
    Image.fromarray(rgb).save(srgb_path)


if __name__ == "__main__":
    unittest.main()
