from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from perception_isp.pascalraw_dataset import (
    build_pascalraw_image_availability,
    build_pascalraw_subset_plan,
    extract_pascalraw_images,
    load_pascalraw_annotation_records,
    main as pascalraw_dataset_main,
    write_pascalraw_image_availability,
    write_pascalraw_subset_plan,
)


class PascalRawDatasetTest(unittest.TestCase):
    def test_load_pascalraw_annotation_records_parses_voc_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ann = Path(tmp) / "Annotations"
            ann.mkdir()
            (ann / "2014_000001.xml").write_text(_voc_xml("2014_000001.png", label="Person"))

            records = load_pascalraw_annotation_records(ann)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].sample_id, "2014_000001")
            self.assertEqual(records[0].image_file_name, "2014_000001.png")
            self.assertEqual(records[0].width, 120)
            self.assertEqual(records[0].height, 90)
            self.assertEqual(records[0].boxes[0].label, "person")
            self.assertEqual(records[0].boxes[0].xyxy, (10.0, 12.0, 50.0, 60.0))

    def test_subset_plan_maps_expected_zip_and_extracted_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ann = Path(tmp) / "Annotations"
            ann.mkdir()
            (ann / "2014_000001.xml").write_text(_voc_xml("2014_000001.png", label="car"))
            (ann / "2014_000002.xml").write_text(_voc_xml("2014_000002.png", label="bicycle"))

            summary = build_pascalraw_subset_plan(ann, max_total=1)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_count"], 1)
            row = summary["manifest"][0]
            self.assertEqual(row["expected_image_relative_path"], "images/2014_000001.png")
            self.assertEqual(row["expected_zip_member"], "2014_000001.png")
            self.assertEqual(row["labels"], ["car"])

    def test_availability_and_extract_use_zip_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = [
                {
                    "sample_id": "2014_000001",
                    "selection_condition": "daylight_raw_derived_downsampled",
                    "box_count": 1,
                    "image_file_name": "2014_000001.png",
                    "expected_zip_member": "2014_000001.png",
                    "expected_image_relative_path": "images/2014_000001.png",
                }
            ]
            with zipfile.ZipFile(root / "JPEGImages.zip", "w") as archive:
                archive.writestr("2014_000001.png", b"png")

            before = build_pascalraw_image_availability(manifest, dataset_root=root)
            self.assertEqual(before["status"], "extractable")
            self.assertFalse(before["evaluation_ready"])
            self.assertTrue(before["ready_to_extract"])

            extracted = extract_pascalraw_images(manifest, dataset_root=root)
            self.assertEqual(extracted["status"], "pass")
            self.assertTrue((root / "images" / "2014_000001.png").exists())

            after = build_pascalraw_image_availability(manifest, dataset_root=root)
            self.assertEqual(after["status"], "pass")
            self.assertTrue(after["evaluation_ready"])

    def test_writers_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann = root / "Annotations"
            ann.mkdir()
            (ann / "2014_000001.xml").write_text(_voc_xml("2014_000001.png", label="person"))

            summary = build_pascalraw_subset_plan(ann, max_total=1)
            subset_html = write_pascalraw_subset_plan(summary, root / "subset")
            availability_html = write_pascalraw_image_availability(
                build_pascalraw_image_availability(summary["manifest"], dataset_root=root),
                root / "availability",
            )
            self.assertTrue((subset_html.parent / "pascalraw_subset_manifest.json").exists())
            self.assertTrue((availability_html.parent / "pascalraw_missing_files.txt").exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = pascalraw_dataset_main(["plan", str(ann), "--max-total", "1", "--output-dir", str(root / "cli")])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["selected_count"], 1)
            self.assertTrue((root / "cli" / "pascalraw_subset_plan_summary.json").exists())


def _voc_xml(filename: str, *, label: str) -> str:
    return f"""<annotation>
  <folder>PASCALRAW</folder>
  <filename>{filename}</filename>
  <size><width>120</width><height>90</height><depth>3</depth></size>
  <object>
    <name>{label}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox><xmin>10</xmin><ymin>12</ymin><xmax>50</xmax><ymax>60</ymax></bndbox>
  </object>
</annotation>
"""


if __name__ == "__main__":
    unittest.main()
