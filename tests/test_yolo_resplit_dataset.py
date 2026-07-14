from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.datasets.yolo_resplit_dataset import create_resplit_dataset


class YoloResplitDatasetTest(unittest.TestCase):
    def test_creates_resplit_manifest_and_hard_train_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = _make_yolo_source(Path(tmp) / "source", channels=3)
            destination = Path(tmp) / "resplit"
            manifest = Path(tmp) / "split_manifest.json"

            summary = create_resplit_dataset(
                source=source,
                destination=destination,
                seed=7,
                eval_fraction=0.4,
                write_split_manifest=manifest,
                hard_repeat=3,
                copy=True,
            )

            self.assertTrue(manifest.is_file())
            self.assertEqual(summary["source_item_count"], 5)
            self.assertEqual(summary["val"]["source_images"], 2)
            self.assertEqual(summary["train"]["source_images"], 3)
            self.assertGreaterEqual(summary["train"]["output_images"], summary["train"]["source_images"])
            self.assertIn("channels: 3", (destination / "data.yaml").read_text())
            self.assertEqual(len(list((destination / "labels" / "val").glob("*_harddup*.txt"))), 0)

    def test_reuses_manifest_for_matched_rgb_aux_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rgb_source = _make_yolo_source(Path(tmp) / "rgb", channels=3)
            aux_source = _make_yolo_source(Path(tmp) / "aux", channels=6)
            manifest = Path(tmp) / "split_manifest.json"

            rgb_summary = create_resplit_dataset(
                source=rgb_source,
                destination=Path(tmp) / "rgb_resplit",
                seed=11,
                eval_fraction=0.4,
                write_split_manifest=manifest,
                hard_repeat=2,
                copy=True,
            )
            aux_summary = create_resplit_dataset(
                source=aux_source,
                destination=Path(tmp) / "aux_resplit",
                split_manifest=manifest,
                hard_repeat=2,
                copy=True,
            )

            self.assertEqual(rgb_summary["val_sample_ids"], aux_summary["val_sample_ids"])
            self.assertEqual(rgb_summary["train_sample_ids"], aux_summary["train_sample_ids"])
            self.assertEqual(aux_summary["channels"], 6)
            self.assertIn("channels: 6", (Path(tmp) / "aux_resplit" / "data.yaml").read_text())


def _make_yolo_source(root: Path, *, channels: int) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    names = ["bicycle", "car", "person"]
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nchannels: {channels}\nnames: {names}\n"
    )
    sample_ids = ["2014_000001", "2014_000002", "2014_000003", "2014_000004", "2014_000005"]
    for index, sample_id in enumerate(sample_ids):
        split = "train" if index < 3 else "val"
        stem = f"{index:05d}_{sample_id}"
        image = np.zeros((12, 16, min(channels, 3)), dtype=np.uint8)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        Image.fromarray(image[:, :, :3]).save(root / "images" / split / f"{stem}.png")
        if channels > 3:
            np.save(root / "images" / split / f"{stem}.npy", np.zeros((12, 16, channels), dtype=np.uint8))
        label = "0 0.5 0.5 0.05 0.05\n" if index % 2 == 0 else "1 0.5 0.5 0.4 0.2\n"
        (root / "labels" / split / f"{stem}.txt").write_text(label)
    return root


if __name__ == "__main__":
    unittest.main()
