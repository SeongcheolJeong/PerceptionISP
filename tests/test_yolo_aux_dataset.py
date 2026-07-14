from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from perception_isp.aux_export import EDGE6_CHANNELS, export_aux_dataset
from perception_isp.synthetic_eval import make_synthetic_evaluation_samples
from perception_isp.yolo_aux_dataset import export_yolo_aux_dataset


class YoloAuxDatasetTest(unittest.TestCase):
    def test_exports_rgb_aux_yolo_npy_dataset(self) -> None:
        samples = make_synthetic_evaluation_samples(count=4, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "aux_export"
            export_aux_dataset(samples, export_root, include_extended=True, include_preview=False, compress=False)

            summary = export_yolo_aux_dataset(
                manifest_path=export_root / "manifest.jsonl",
                output_dir=Path(tmp) / "yolo_aux",
                tensor_key="rgb_aux_extended_hwc",
                channel_mode="rgb_aux",
                include_labels=("car", "person"),
                eval_fraction=0.25,
            )

            root = Path(summary["output_dir"])
            self.assertEqual(summary["channels"], 16)
            self.assertEqual(summary["train_count"], 3)
            self.assertEqual(summary["val_count"], 1)
            self.assertIn("channels: 16", (root / "data.yaml").read_text())
            npy_files = sorted((root / "images").rglob("*.npy"))
            self.assertEqual(len(npy_files), 4)
            arr = np.load(npy_files[0])
            self.assertEqual(arr.shape, (32, 48, 16))
            self.assertEqual(arr.dtype, np.uint8)
            label_files = sorted((root / "labels").rglob("*.txt"))
            self.assertEqual(len(label_files), 4)
            self.assertTrue(any(path.read_text().strip() for path in label_files))

    def test_exports_rgb_only_with_three_channels(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=40, height=24)
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "aux_export"
            export_aux_dataset(samples, export_root, include_extended=True, include_preview=False, compress=False)

            summary = export_yolo_aux_dataset(
                manifest_path=export_root / "manifest.jsonl",
                output_dir=Path(tmp) / "yolo_rgb",
                tensor_key="rgb_aux_extended_chw",
                channel_mode="rgb_only",
                include_labels=("car", "person"),
                eval_fraction=0.34,
            )

            root = Path(summary["output_dir"])
            self.assertEqual(summary["channels"], 3)
            self.assertIn("channels: 3", (root / "data.yaml").read_text())
            arr = np.load(sorted((root / "images").rglob("*.npy"))[0])
            self.assertEqual(arr.shape, (24, 40, 3))

    def test_exports_selected_rgb_aux_channels(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=40, height=24)
        selected = (
            "rgb_r",
            "rgb_g",
            "rgb_b",
            "aux_edge_strength",
            "aux_edge_evidence",
            "aux_psf_edge_likelihood",
        )
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "aux_export"
            export_aux_dataset(samples, export_root, include_extended=True, include_preview=False, compress=False)

            summary = export_yolo_aux_dataset(
                manifest_path=export_root / "manifest.jsonl",
                output_dir=Path(tmp) / "yolo_selected",
                tensor_key="rgb_aux_extended_hwc",
                channel_mode="rgb_aux",
                selected_channels=selected,
                include_labels=("car", "person"),
                eval_fraction=0.34,
            )

            root = Path(summary["output_dir"])
            self.assertEqual(summary["channels"], 6)
            self.assertEqual(tuple(summary["selected_channels"]), selected)
            self.assertEqual(tuple(summary["channel_names"]), selected)
            self.assertIn("channels: 6", (root / "data.yaml").read_text())
            arr = np.load(sorted((root / "images").rglob("*.npy"))[0])
            self.assertEqual(arr.shape, (24, 40, 6))

    def test_uses_npz_channel_names_for_custom_base_tensor(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=40, height=24)
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "aux_export"
            export_aux_dataset(
                samples,
                export_root,
                channels=EDGE6_CHANNELS,
                include_extended=False,
                include_preview=False,
                compress=False,
            )

            summary = export_yolo_aux_dataset(
                manifest_path=export_root / "manifest.jsonl",
                output_dir=Path(tmp) / "yolo_edge6",
                tensor_key="rgb_aux_hwc",
                channel_mode="rgb_aux",
                include_labels=("car", "person"),
                eval_fraction=0.34,
            )

            self.assertEqual(summary["channels"], len(EDGE6_CHANNELS))
            self.assertEqual(tuple(summary["channel_names"]), EDGE6_CHANNELS)
            self.assertIn("aux_edge_evidence", summary["channel_names"])
            root = Path(summary["output_dir"])
            arr = np.load(sorted((root / "images").rglob("*.npy"))[0])
            self.assertEqual(arr.shape, (24, 40, len(EDGE6_CHANNELS)))

    def test_drops_labels_outside_include_list(self) -> None:
        samples = make_synthetic_evaluation_samples(count=2, width=40, height=24)
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "aux_export"
            export_aux_dataset(samples, export_root, include_extended=True, include_preview=False, compress=False)
            first_label = next((export_root / "labels").glob("*.json"))
            payload = json.loads(first_label.read_text())
            payload["labels"] = ["ignored_class"]
            payload["boxes"] = payload["boxes"][:1]
            payload["boxes_xyxy"] = payload["boxes_xyxy"][:1]
            payload["boxes_xyxy_normalized"] = payload["boxes_xyxy_normalized"][:1]
            first_label.write_text(json.dumps(payload))

            summary = export_yolo_aux_dataset(
                manifest_path=export_root / "manifest.jsonl",
                output_dir=Path(tmp) / "yolo_filtered",
                include_labels=("car",),
                eval_fraction=0.5,
            )

            self.assertGreaterEqual(sum(item["dropped_boxes"] for item in summary["exported"]), 1)


if __name__ == "__main__":
    unittest.main()
