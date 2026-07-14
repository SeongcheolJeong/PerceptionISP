from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from perception_isp.datasets.lod_dataset import (
    build_lod_image_availability,
    build_lod_local_readiness,
    build_lod_subset_plan,
    load_lod_annotation_records,
    main as lod_dataset_main,
    write_lod_image_availability,
    write_lod_local_readiness,
    write_lod_subset_plan,
)


class LODDatasetTest(unittest.TestCase):
    def test_load_lod_annotation_records_parses_voc_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann = root / "ann"
            ann.mkdir()
            (ann / "000001.xml").write_text(_voc_xml("000001.JPG", label="Person"))

            records = load_lod_annotation_records(ann)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].sample_id, "000001")
            self.assertEqual(records[0].image_file_name, "000001.JPG")
            self.assertEqual(records[0].width, 100)
            self.assertEqual(records[0].height, 80)
            self.assertEqual(records[0].boxes[0].label, "person")
            self.assertEqual(records[0].boxes[0].xyxy, (10.0, 12.0, 50.0, 60.0))

    def test_build_subset_plan_maps_expected_lod_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ann = Path(tmp) / "RAW-dark-Annotations"
            ann.mkdir()
            (ann / "000001.xml").write_text(_voc_xml("000001.JPG", label="car"))
            (ann / "000002.xml").write_text(_voc_xml("000002.JPG", label="bicycle"))

            summary = build_lod_subset_plan(ann, max_total=1)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_count"], 1)
            row = summary["manifest"][0]
            self.assertEqual(row["expected_raw_relative_path"], "RAW-dark-images/000001.CR2")
            self.assertEqual(row["expected_srgb_relative_path"], "RGB-dark-images/000001.JPG")
            self.assertEqual(row["labels"], ["car"])

    def test_availability_audit_counts_missing_raw_and_srgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = [
                {
                    "sample_id": "000001",
                    "selection_condition": "low_light",
                    "box_count": 1,
                    "expected_raw_relative_path": "RAW-dark-images/000001.CR2",
                    "expected_srgb_relative_path": "RGB-dark-images/000001.JPG",
                },
                {
                    "sample_id": "000002",
                    "selection_condition": "low_light",
                    "box_count": 1,
                    "expected_raw_relative_path": "RAW-dark-images/000002.CR2",
                    "expected_srgb_relative_path": "RGB-dark-images/000002.JPG",
                },
            ]
            _touch(root / "RAW-dark-images" / "000001.CR2")

            summary = build_lod_image_availability(manifest, dataset_root=root)

            self.assertEqual(summary["status"], "blocked")
            self.assertFalse(summary["evaluation_ready"])
            self.assertEqual(summary["required_file_count"], 4)
            self.assertEqual(summary["available_file_count"], 1)
            self.assertEqual(summary["missing_raw_count"], 1)
            self.assertEqual(summary["missing_srgb_count"], 2)

    def test_local_readiness_blocks_without_downloaded_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_lod_local_readiness(dataset_root=Path(tmp) / "lod")

            self.assertEqual(summary["status"], "blocked")
            self.assertFalse(summary["ready_for_subset_plan"])
            self.assertFalse(summary["ready_for_image_eval"])
            checks = {row["id"]: row["status"] for row in summary["checks"]}
            self.assertEqual(checks["voc_annotations_present"], "fail")

    def test_zip_plan_writers_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "lod_ann.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("RAW-dark-Annotations/000001.xml", _voc_xml("000001.JPG", label="person"))
            summary = build_lod_subset_plan(archive, max_total=1)
            subset_html = write_lod_subset_plan(summary, root / "subset")
            availability_html = write_lod_image_availability(build_lod_image_availability(summary["manifest"], dataset_root=root), root / "avail")
            readiness_html = write_lod_local_readiness(build_lod_local_readiness(dataset_root=root), root / "ready")
            self.assertTrue((subset_html.parent / "lod_subset_manifest.json").exists())
            self.assertTrue((availability_html.parent / "lod_missing_files.txt").exists())
            self.assertTrue((readiness_html.parent / "lod_local_readiness_summary.json").exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = lod_dataset_main(["plan", str(archive), "--max-total", "1", "--output-dir", str(root / "cli")])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["selected_count"], 1)
            self.assertTrue((root / "cli" / "lod_subset_plan_summary.json").exists())


def _voc_xml(filename: str, *, label: str) -> str:
    return f"""<annotation>
  <folder>LOD</folder>
  <filename>{filename}</filename>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object>
    <name>{label}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox><xmin>10</xmin><ymin>12</ymin><xmax>50</xmax><ymax>60</ymax></bndbox>
  </object>
</annotation>
"""


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unit")


if __name__ == "__main__":
    unittest.main()
