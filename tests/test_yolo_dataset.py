from __future__ import annotations

import tempfile
import unittest
import os
from contextlib import redirect_stderr
from io import StringIO
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

    def test_load_progress_interval_writes_status_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            for index in range(2):
                Image.fromarray(rgb).save(image_dir / f"sample_{index}.jpg")
                (label_dir / f"sample_{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
            (root / "data.yaml").write_text("path: .\nval: images/val\nnames: ['car']\n")

            stream = StringIO()
            with redirect_stderr(stream):
                samples = load_yolo_detection_samples(
                    root / "data.yaml",
                    split="val",
                    limit=2,
                    width=64,
                    height=48,
                    use_camerae2e=False,
                    progress_interval=1,
                    progress_label="unit-load",
                )

            self.assertEqual(len(samples), 2)
            self.assertIn("[unit-load] loaded 1/2 samples", stream.getvalue())
            self.assertIn("[unit-load] loaded 2/2 samples", stream.getvalue())

    def test_coco_root_without_yaml_uses_coco_class_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "coco128"
            image_dir = root / "images" / "train2017"
            label_dir = root / "labels" / "train2017"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[8:16, 10:22, :] = 255
            Image.fromarray(rgb).save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n")

            samples = load_yolo_detection_samples(
                root,
                split="train2017",
                limit=1,
                width=64,
                height=48,
                use_camerae2e=False,
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].ground_truth[0].label, "person")

    def test_relative_root_without_yaml_does_not_duplicate_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "coco128"
            image_dir = root / "images" / "train2017"
            label_dir = root / "labels" / "train2017"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n")
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                samples = load_yolo_detection_samples(
                    Path("coco128"),
                    split="train2017",
                    limit=1,
                    width=64,
                    height=48,
                    use_camerae2e=False,
                )
            finally:
                os.chdir(previous)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].metadata["image_path"].count("coco128"), 1)

    def test_data_yaml_inside_named_root_does_not_duplicate_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kitti"
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n")
            (root / "data.yaml").write_text("path: kitti\nval: images/val\nnames: ['car']\n")

            samples = load_yolo_detection_samples(
                root / "data.yaml",
                split="val",
                limit=1,
                width=64,
                height=48,
                use_camerae2e=False,
            )

            self.assertEqual(len(samples), 1)
            self.assertNotIn("kitti/kitti", samples[0].metadata["image_path"])

    def test_kitti_root_without_yaml_names_uses_kitti_class_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kitti"
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("3 0.5 0.5 0.25 0.25\n")

            samples = load_yolo_detection_samples(
                root,
                split="val",
                limit=1,
                width=64,
                height=48,
                use_camerae2e=False,
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].ground_truth[0].label, "pedestrian")

    def test_offset_skips_images_before_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "val"
            label_dir = root / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            for name, class_id in (("a", 0), ("b", 1)):
                Image.fromarray(rgb).save(image_dir / f"{name}.jpg")
                (label_dir / f"{name}.txt").write_text(f"{class_id} 0.5 0.5 0.25 0.25\n")
            (root / "data.yaml").write_text("path: .\nval: images/val\nnames: ['car', 'person']\n")

            samples = load_yolo_detection_samples(
                root / "data.yaml",
                split="val",
                offset=1,
                limit=1,
                width=64,
                height=48,
                use_camerae2e=False,
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "b")
            self.assertEqual(samples[0].metadata["dataset_index"], 1)
            self.assertEqual(samples[0].ground_truth[0].label, "person")


if __name__ == "__main__":
    unittest.main()
