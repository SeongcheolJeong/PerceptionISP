from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from perception_isp.aodraw_annotation_audit import (
    build_aodraw_annotation_audit,
    main as aodraw_annotation_audit_main,
    write_aodraw_annotation_audit,
)


class AODRawAnnotationAuditTest(unittest.TestCase):
    def test_build_audit_summarizes_coco_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = _write_zip(root / "AODRaw_annotations.zip")

            summary = build_aodraw_annotation_audit(archive)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["recommended_first_split"], "test_annotations_downsample_scale3_bbox_min_size32.json")
            self.assertEqual(summary["split_count"], 4)
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["required_annotation_jsons_present"], "pass")
            self.assertEqual(checks["adverse_tags_present"], "pass")
            split = next(row for row in summary["splits"] if row["name"] == "test_annotations_downsample_scale3_bbox_min_size32.json")
            self.assertEqual(split["object_category_count"], 2)
            self.assertEqual(split["ignore_category_id"], 2)
            self.assertEqual(split["tags"]["low_light"], 1)

    def test_missing_required_json_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "partial.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("AODRaw/annotations/test_annotations.json", json.dumps(_payload()))

            summary = build_aodraw_annotation_audit(archive)

            self.assertEqual(summary["status"], "warning")
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["required_annotation_jsons_present"], "fail")

    def test_ignored_zero_area_boxes_do_not_fail_schema_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "crowd.zip"
            payload = _payload()
            payload["annotations"].append(
                {"id": 12, "image_id": 2, "category_id": 1, "bbox": [10, 20, 0, 30], "area": 0, "iscrowd": 1}
            )
            with zipfile.ZipFile(archive, "w") as z:
                for name in (
                    "train_annotations.json",
                    "test_annotations.json",
                    "train_annotations_downsample_scale3_bbox_min_size32.json",
                    "test_annotations_downsample_scale3_bbox_min_size32.json",
                ):
                    z.writestr(f"AODRaw/annotations/{name}", json.dumps(payload))

            summary = build_aodraw_annotation_audit(archive)

            checks = {row["id"]: row for row in summary["checks"]}
            self.assertEqual(checks["bbox_schema_valid"]["status"], "pass")
            self.assertIn("ignored_invalid_bbox_count=4", checks["bbox_schema_valid"]["evidence"])

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = _write_zip(root / "AODRaw_annotations.zip")
            summary = build_aodraw_annotation_audit(archive)
            html_path = write_aodraw_annotation_audit(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_annotation_audit_summary.json").exists())
            self.assertIn("AODRaw Annotation Audit", html_path.read_text())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aodraw_annotation_audit_main([str(archive), "--output-dir", str(root / "cli")])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "pass")
            self.assertEqual(printed["recommended_first_split"], "test_annotations_downsample_scale3_bbox_min_size32.json")


def _write_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name in (
            "train_annotations.json",
            "test_annotations.json",
            "train_annotations_downsample_scale3_bbox_min_size32.json",
            "test_annotations_downsample_scale3_bbox_min_size32.json",
        ):
            z.writestr(f"AODRaw/annotations/{name}", json.dumps(_payload()))
    return path


def _payload() -> dict:
    return {
        "images": [
            {"id": 1, "file_name": "00000001.JPG", "width": 2000, "height": 1333, "tag": ["low_light"]},
            {"id": 2, "file_name": "00000002.JPG", "width": 2000, "height": 1333, "tag": ["rain", "fog"]},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 0, "bbox": [1, 2, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 11, "image_id": 2, "category_id": 1, "bbox": [3, 4, 50, 60], "area": 3000, "iscrowd": 0},
        ],
        "categories": [
            {"id": 0, "name": "person", "supercategory": "none"},
            {"id": 1, "name": "car", "supercategory": "none"},
            {"id": 2, "name": "ignore", "supercategory": "none"},
        ],
    }


if __name__ == "__main__":
    unittest.main()
