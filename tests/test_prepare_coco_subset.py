from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.prepare_coco_subset import prepare_coco_val_subset


class PrepareCocoSubsetTest(unittest.TestCase):
    def test_prepare_subset_from_local_label_zip_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_zip = root / "coco2017labels.zip"
            with zipfile.ZipFile(labels_zip, "w") as archive:
                archive.writestr("coco/val2017.txt", "images/val2017/000000000001.jpg\nimages/val2017/000000000002.jpg\n")
                archive.writestr("coco/labels/val2017/000000000001.txt", "0 0.5 0.5 0.25 0.25\n")

            image_source = root / "source_images"
            image_source.mkdir()
            for name in ("000000000001.jpg", "000000000002.jpg"):
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_source / name)

            output_dir = root / "coco_val2017_2"
            summary = prepare_coco_val_subset(
                output_dir=output_dir,
                count=2,
                labels_zip=labels_zip,
                cache_dir=root / "cache",
                image_source_dir=image_source,
                threads=2,
            )

            self.assertEqual(summary["image_count"], 2)
            self.assertEqual(summary["label_file_count"], 1)
            self.assertEqual(summary["empty_label_file_count"], 1)
            self.assertTrue((output_dir / "images" / "val2017" / "000000000001.jpg").exists())
            self.assertTrue((output_dir / "labels" / "val2017" / "000000000001.txt").exists())
            self.assertEqual((output_dir / "labels" / "val2017" / "000000000002.txt").read_text(), "")
            self.assertIn("person", (output_dir / "data.yaml").read_text())
            self.assertEqual(json.loads((output_dir / "prepare_summary.json").read_text())["image_count"], 2)


if __name__ == "__main__":
    unittest.main()
