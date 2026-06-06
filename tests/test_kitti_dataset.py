from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.kitti_dataset import load_kitti_detection_samples


class KittiDatasetAdapterTest(unittest.TestCase):
    def test_loads_native_kitti_labels_with_direct_raw_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "training" / "image_2"
            label_dir = root / "training" / "label_2"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[8:16, 10:22, :] = 255
            Image.fromarray(rgb).save(image_dir / "000001.png")
            (label_dir / "000001.txt").write_text(
                "Car 0.00 0 -1.57 10.0 8.0 22.0 16.0 1.5 1.6 4.0 0.0 0.0 0.0 0.0\n"
                "DontCare -1 -1 -10 0.0 0.0 2.0 2.0 -1 -1 -1 -1000 -1000 -1000 -10\n"
            )

            samples = load_kitti_detection_samples(
                root,
                split="training",
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
            self.assertEqual(samples[0].ground_truth[0].xyxy, (20.0, 16.0, 44.0, 32.0))


if __name__ == "__main__":
    unittest.main()
