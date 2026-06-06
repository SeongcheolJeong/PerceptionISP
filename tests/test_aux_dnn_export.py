from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from perception_isp.aux_dnn import RGB_AUX_CHANNELS, build_rgb_aux_tensor, labels_from_manifest, make_aux_early_fusion_stem, make_torch_dataset
from perception_isp.aux_eval_dense import evaluate_dense_manifest
from perception_isp.aux_export import _load_samples, export_aux_dataset
from perception_isp.aux_train_dense import train_dense
from perception_isp.aux_train_smoke import train_smoke
from perception_isp.comparison import build_pipeline_images, compare_dataset
from perception_isp.detectors import RGBAuxTorchDenseDetector, RGBAuxTorchSmokeDetector, rgb_aux_detector_from_checkpoint
from perception_isp.synthetic_eval import make_synthetic_evaluation_samples


class AuxDNNExportTest(unittest.TestCase):
    def test_build_rgb_aux_tensor_layouts(self) -> None:
        sample = make_synthetic_evaluation_samples(count=1, width=64, height=48)[0]
        images = build_pipeline_images(sample)
        hwc = build_rgb_aux_tensor(images, layout="hwc")
        chw = build_rgb_aux_tensor(images, layout="chw")
        self.assertEqual(hwc.shape, (48, 64, 6))
        self.assertEqual(chw.shape, (6, 48, 64))
        self.assertEqual(tuple(RGB_AUX_CHANNELS[:3]), ("rgb_r", "rgb_g", "rgb_b"))
        self.assertTrue(np.isfinite(hwc).all())
        self.assertGreater(float(np.mean(hwc[:, :, 3])), 0.0)

    def test_export_aux_dataset_writes_manifest_tensors_and_labels(self) -> None:
        samples = make_synthetic_evaluation_samples(count=2, width=64, height=48)
        with tempfile.TemporaryDirectory() as tmp:
            summary = export_aux_dataset(samples, tmp)
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            self.assertEqual(summary["sample_count"], 2)
            self.assertGreater(summary["elapsed_seconds"], 0.0)
            self.assertGreater(summary["samples_per_second"], 0.0)
            self.assertGreater(summary["seconds_per_sample"], 0.0)
            self.assertTrue(manifest_path.exists())
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            tensor_path = root / rows[0]["tensor_path"]
            label_path = root / rows[0]["label_path"]
            self.assertTrue(tensor_path.exists())
            self.assertTrue(label_path.exists())
            with np.load(tensor_path) as payload:
                self.assertEqual(payload["rgb_aux_chw"].shape, (6, 48, 64))
                self.assertEqual(payload["rgb_aux_hwc"].shape, (48, 64, 6))
                self.assertEqual(payload["boxes_xyxy"].shape[1], 4)
            labels = json.loads(label_path.read_text())
            self.assertGreater(labels["box_count"], 0)
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "index.html").exists())

    def test_load_samples_for_yolo_dataset_forwards_cache_and_progress_options(self) -> None:
        with mock.patch("perception_isp.yolo_dataset.load_yolo_detection_samples", return_value=()) as loader:
            samples = _load_samples(
                source="yolo-dataset",
                dataset="data/unit.yaml",
                split="val",
                count=3,
                offset=5,
                width=64,
                height=48,
                cfa_pattern="auto",
                use_camerae2e=True,
                progress_interval=2,
                cache_dir=Path("cache/raw"),
            )

        self.assertEqual(samples, ())
        kwargs = loader.call_args.kwargs
        self.assertEqual(kwargs["limit"], 3)
        self.assertEqual(kwargs["offset"], 5)
        self.assertEqual(kwargs["progress_interval"], 2)
        self.assertEqual(kwargs["progress_label"], "load:yolo-dataset:5+3")
        self.assertEqual(kwargs["cache_dir"], Path("cache/raw"))

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_torch_dataset_and_stem_consume_exported_tensors(self) -> None:
        import torch

        samples = make_synthetic_evaluation_samples(count=1, width=64, height=48)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp)
            dataset = make_torch_dataset(Path(tmp) / "manifest.jsonl")
            tensor, target = dataset[0]
            self.assertEqual(tuple(tensor.shape), (6, 48, 64))
            self.assertIn("boxes_normalized", target)
            stem = make_aux_early_fusion_stem(in_channels=6, out_channels=8)
            out = stem(tensor.unsqueeze(0))
            self.assertEqual(out.shape[1], 8)
            self.assertTrue(torch.isfinite(out).all())

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_smoke_training_loop_runs_on_exported_tensors(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp)
            summary = train_smoke(
                manifest_path=Path(tmp) / "manifest.jsonl",
                epochs=1,
                device_name="cpu",
                estimate_samples=(2, 20),
                eval_fraction=0.34,
                output_dir=Path(tmp) / "train",
            )
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["train_sample_count"], 2)
            self.assertEqual(summary["eval_sample_count"], 1)
            self.assertEqual(summary["epochs"], 1)
            self.assertTrue(np.isfinite(summary["last_loss"]))
            self.assertIsNotNone(summary["initial_eval_loss"])
            self.assertIsNotNone(summary["final_eval_loss"])
            self.assertTrue(np.isfinite(summary["history"][0]["eval_loss"]))
            self.assertGreater(summary["elapsed_seconds"], 0.0)
            self.assertGreater(summary["sample_epochs_per_second"], 0.0)
            self.assertEqual(summary["time_estimates"][0]["samples"], 2)
            self.assertEqual(summary["time_estimates"][1]["samples"], 20)
            self.assertTrue((Path(tmp) / "train" / "train_smoke_summary.json").exists())
            self.assertTrue((Path(tmp) / "train" / "rgb_aux_smoke_detector.pt").exists())

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_trained_rgb_aux_smoke_detector_integrates_with_comparison(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp)
            summary = train_smoke(
                manifest_path=Path(tmp) / "manifest.jsonl",
                epochs=1,
                device_name="cpu",
                eval_fraction=0.34,
                output_dir=Path(tmp) / "train",
            )
            detector = RGBAuxTorchSmokeDetector(summary["checkpoint"], confidence=0.0)
            result = compare_dataset(
                samples[:1],
                rgb_aux_detector=detector,
                label_agnostic=True,
                include_images=True,
            )
            self.assertIn("perception_rgb_aux_dnn", result["aggregate"])
            self.assertIn("perception_rgb_aux_dnn", result["samples"][0]["metrics"])
            self.assertIn("perception_rgb_aux_dnn", result["samples"][0]["_visuals"])

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_dense_rgb_aux_detector_trains_and_integrates_with_comparison(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp)
            manifest = Path(tmp) / "manifest.jsonl"
            self.assertEqual(labels_from_manifest(manifest), ("car", "person", "traffic_light"))
            summary = train_dense(
                manifest_path=manifest,
                epochs=1,
                device_name="cpu",
                grid_size=(4, 6),
                base_channels=8,
                eval_fraction=0.34,
                include_labels=("car", "person"),
                estimate_samples=(3, 30),
                output_dir=Path(tmp) / "dense",
            )
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["include_labels"], ["car", "person"])
            self.assertEqual(summary["class_names"], ["car", "person"])
            self.assertEqual(summary["split_strategy"], "coverage")
            self.assertIn("train_indices", summary)
            self.assertIn("eval_indices", summary)
            self.assertIn("checkpoint_epoch", summary)
            self.assertIn("checkpoint_loss", summary)
            self.assertIn("checkpoint_loss_kind", summary)
            self.assertEqual(summary["box_encoding"], "cell_center_size")
            self.assertIn("train_class_names", summary)
            self.assertIn("eval_class_names", summary)
            self.assertIn("missing_eval_class_names", summary)
            self.assertEqual(summary["missing_eval_class_names"], [])
            self.assertGreater(summary["sample_epochs_per_second"], 0.0)
            self.assertTrue((Path(tmp) / "dense" / "rgb_aux_dense_detector.pt").exists())
            self.assertTrue((Path(tmp) / "dense" / "rgb_aux_dense_detector_final.pt").exists())
            self.assertIn("final_checkpoint", summary)
            detector = rgb_aux_detector_from_checkpoint(summary["checkpoint"], confidence=0.0)
            self.assertIsInstance(detector, RGBAuxTorchDenseDetector)
            result = compare_dataset(
                samples[:1],
                rgb_aux_detector=detector,
                label_agnostic=False,
                include_images=True,
            )
            self.assertIn("perception_rgb_aux_dnn", result["aggregate"])
            self.assertIn("perception_rgb_aux_dnn", result["samples"][0]["metrics"])
            self.assertIn("perception_rgb_aux_dnn", result["samples"][0]["_visuals"])
            direct = evaluate_dense_manifest(
                manifest_path=manifest,
                checkpoint_path=summary["checkpoint"],
                split="eval",
                confidence=0.0,
                label_agnostic=False,
                output_dir=Path(tmp) / "dense_eval",
            )
            self.assertEqual(direct["sample_count"], len(summary["eval_indices"]))
            self.assertEqual(direct["eval_labels"], ["car", "person"])
            self.assertTrue(all(sample["gt_count"] == 2 for sample in direct["samples"]))
            self.assertIn("aggregate", direct)
            self.assertTrue((Path(tmp) / "dense_eval" / "dense_eval_summary.json").exists())
            self.assertTrue((Path(tmp) / "dense_eval" / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
