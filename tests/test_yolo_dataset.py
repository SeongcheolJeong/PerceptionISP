from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.yolo_dataset import load_yolo_detection_samples


class YoloDatasetAdapterTest(unittest.TestCase):
    def test_loads_yolo_dataset_with_direct_raw_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[8:16, 10:22, :] = 255
            Image.fromarray(rgb).save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n")
            (root / "data.yaml").write_text("path: .\nval: images/val\nnames: ['car']\n")

            samples = load_yolo_detection_samples(
                root / "data.yaml",
                split="val",
                limit=1,
                width=64,
                height=48,
                use_camerae2e=False,
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].raw.data.shape, (3, 48, 64))
            self.assertIsNotNone(samples[0].reference_rgb)
            self.assertEqual(samples[0].reference_rgb.shape, (48, 64, 3))
            self.assertEqual(len(samples[0].ground_truth), 1)
            self.assertEqual(samples[0].ground_truth[0].label, "car")


if __name__ == "__main__":
    unittest.main()
