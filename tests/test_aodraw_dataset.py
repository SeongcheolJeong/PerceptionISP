from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from perception_isp.aodraw_dataset import (
    build_aodraw_subset_plan,
    load_aodraw_annotation_records,
    main as aodraw_dataset_main,
    write_aodraw_subset_plan,
)


class AODRawDatasetTest(unittest.TestCase):
    def test_load_records_skips_ignore_and_crowd_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = _write_zip(Path(tmp) / "ann.zip")

            records = load_aodraw_annotation_records(archive, split="test_annotations_downsample_scale3_bbox_min_size32.json")

            self.assertEqual(len(records), 4)
            low = next(row for row in records if row.file_name == "00000001.JPG")
            self.assertEqual(low.tags, ("low_light",))
            self.assertEqual(len(low.boxes), 1)
            self.assertEqual(low.ignored_annotation_count, 2)
            self.assertEqual(low.boxes[0].xyxy, (10.0, 20.0, 40.0, 60.0))
            self.assertEqual(low.boxes[0].label, "person")

    def test_build_subset_plan_maps_downsample_raw_to_npy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = _write_zip(Path(tmp) / "ann.zip")

            summary = build_aodraw_subset_plan(
                archive,
                split="test_annotations_downsample_scale3_bbox_min_size32.json",
                conditions=("low_light", "rain", "fog"),
                per_condition=1,
                max_total=3,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_count"], 3)
            self.assertEqual(summary["condition_counts"], {"low_light": 1, "rain": 1, "fog": 1})
            manifest = summary["manifest"]
            self.assertTrue(all(row["raw_file_name"].endswith(".npy") for row in manifest))
            self.assertEqual(manifest[0]["expected_raw_relative_path"], "images_downsampled_raw/00000001.npy")
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["selected_rows_have_boxes"], "pass")

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = _write_zip(root / "ann.zip")
            summary = build_aodraw_subset_plan(archive, per_condition=1, max_total=3)
            html_path = write_aodraw_subset_plan(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_subset_plan_summary.json").exists())
            self.assertTrue((html_path.parent / "aodraw_subset_manifest.json").exists())
            self.assertIn("AODRaw Subset Plan", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aodraw_dataset_main(
                    [
                        str(archive),
                        "--per-condition",
                        "1",
                        "--max-total",
                        "3",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["selected_count"], 3)
            self.assertTrue((root / "cli" / "aodraw_subset_manifest.json").exists())


def _write_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("AODRaw/annotations/test_annotations_downsample_scale3_bbox_min_size32.json", json.dumps(_payload()))
    return path


def _payload() -> dict:
    images = [
        {"id": 1, "file_name": "00000001.JPG", "width": 2000, "height": 1333, "tag": ["low_light"]},
        {"id": 2, "file_name": "00000002.JPG", "width": 2000, "height": 1333, "tag": ["rain"]},
        {"id": 3, "file_name": "00000003.JPG", "width": 2000, "height": 1333, "tag": ["fog"]},
        {"id": 4, "file_name": "00000004.JPG", "width": 2000, "height": 1333, "tag": ["normal_light"]},
    ]
    annotations = [
        {"id": 10, "image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0, "ignore": 0},
        {"id": 11, "image_id": 1, "category_id": 1, "bbox": [1, 1, 0, 5], "area": 0, "iscrowd": 1, "ignore": 0},
        {"id": 12, "image_id": 1, "category_id": 2, "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0, "ignore": 0},
        {"id": 20, "image_id": 2, "category_id": 1, "bbox": [15, 25, 35, 45], "area": 1575, "iscrowd": 0, "ignore": 0},
        {"id": 30, "image_id": 3, "category_id": 1, "bbox": [20, 30, 40, 50], "area": 2000, "iscrowd": 0, "ignore": 0},
        {"id": 40, "image_id": 4, "category_id": 0, "bbox": [25, 35, 45, 55], "area": 2475, "iscrowd": 0, "ignore": 0},
    ]
    categories = [
        {"id": 0, "name": "person", "supercategory": "none"},
        {"id": 1, "name": "car", "supercategory": "none"},
        {"id": 2, "name": "ignore", "supercategory": "none"},
    ]
    return {"images": images, "annotations": annotations, "categories": categories}


if __name__ == "__main__":
    unittest.main()
