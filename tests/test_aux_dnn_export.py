from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

from perception_isp.aux_dnn import (
    RGB_AUX_CHANNELS,
    RGB_AUX_EXTENDED_CHANNELS,
    apply_channel_mask,
    build_rgb_aux_extended_tensor,
    build_rgb_aux_tensor,
    channel_mask_for_mode,
    channels_for_tensor_key,
    labels_from_manifest,
    make_aux_early_fusion_stem,
    make_torch_dataset,
)
from perception_isp.aux_eval_dense import _apply_input_ablation, _shuffle_indices, evaluate_dense_manifest
from perception_isp.aux_export import EDGE6_CHANNELS, _load_samples, export_aux_dataset, main as aux_export_main
from perception_isp.aux_train_dense import _dense_targets, _weighted_mean, train_dense
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
        extended = build_rgb_aux_extended_tensor(images, layout="hwc")
        self.assertEqual(hwc.shape, (48, 64, 6))
        self.assertEqual(chw.shape, (6, 48, 64))
        self.assertEqual(extended.shape, (48, 64, len(RGB_AUX_EXTENDED_CHANNELS)))
        self.assertEqual(tuple(RGB_AUX_CHANNELS[:3]), ("rgb_r", "rgb_g", "rgb_b"))
        self.assertIn("aux_demosaic_confidence", RGB_AUX_EXTENDED_CHANNELS)
        self.assertIn("aux_edge_evidence", RGB_AUX_EXTENDED_CHANNELS)
        self.assertIn("aux_psf_edge_likelihood", RGB_AUX_EXTENDED_CHANNELS)
        self.assertIn("clipping_distance", images.aux_maps)
        self.assertIn("edge_evidence", images.aux_maps)
        self.assertIn("psf_edge_likelihood", images.aux_maps)
        self.assertTrue(np.isfinite(hwc).all())
        self.assertTrue(np.isfinite(extended).all())
        self.assertGreater(float(np.mean(hwc[:, :, 3])), 0.0)
        psf_index = RGB_AUX_EXTENDED_CHANNELS.index("aux_psf_edge_likelihood")
        self.assertTrue(np.allclose(extended[:, :, psf_index], images.aux_maps["psf_edge_likelihood"]))
        evidence_index = RGB_AUX_EXTENDED_CHANNELS.index("aux_edge_evidence")
        self.assertTrue(np.allclose(extended[:, :, evidence_index], images.aux_maps["edge_evidence"]))

    def test_channel_masks_zero_expected_groups(self) -> None:
        tensor = np.ones((6, 2, 3), dtype=np.float32)
        rgb_only = apply_channel_mask(tensor, channel_mask_for_mode("rgb_only"))
        aux_only = apply_channel_mask(tensor, channel_mask_for_mode("aux_only"))
        self.assertTrue(np.all(rgb_only[:3] == 1.0))
        self.assertTrue(np.all(rgb_only[3:] == 0.0))
        self.assertTrue(np.all(aux_only[:3] == 0.0))
        self.assertTrue(np.all(aux_only[3:] == 1.0))
        extended_tensor = np.ones((len(RGB_AUX_EXTENDED_CHANNELS), 2, 3), dtype=np.float32)
        extended_aux_only = apply_channel_mask(
            extended_tensor,
            channel_mask_for_mode("aux_only", channels=RGB_AUX_EXTENDED_CHANNELS),
        )
        self.assertTrue(np.all(extended_aux_only[:3] == 0.0))
        self.assertTrue(np.all(extended_aux_only[3:] == 1.0))

    def test_dense_eval_input_ablation_transforms_expected_channels(self) -> None:
        tensor = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
        source = tensor + 100.0

        zero_aux = _apply_input_ablation(tensor, mode="zero_aux")
        self.assertTrue(np.allclose(zero_aux[:, :, :3], tensor[:, :, :3]))
        self.assertTrue(np.allclose(zero_aux[:, :, 3:], 0.0))

        zero_rgb = _apply_input_ablation(tensor, mode="zero_rgb")
        self.assertTrue(np.allclose(zero_rgb[:, :, :3], 0.0))
        self.assertTrue(np.allclose(zero_rgb[:, :, 3:], tensor[:, :, 3:]))

        shuffled_aux = _apply_input_ablation(tensor, mode="shuffle_aux", aux_source_tensor=source)
        self.assertTrue(np.allclose(shuffled_aux[:, :, :3], tensor[:, :, :3]))
        self.assertTrue(np.allclose(shuffled_aux[:, :, 3:], source[:, :, 3:]))

        mapping = _shuffle_indices((1, 2, 3), seed=7)
        self.assertEqual(set(mapping), {1, 2, 3})
        self.assertEqual(set(mapping.values()), {1, 2, 3})

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
                self.assertEqual(payload["rgb_aux_extended_chw"].shape, (len(RGB_AUX_EXTENDED_CHANNELS), 48, 64))
                self.assertEqual(payload["rgb_aux_extended_hwc"].shape, (48, 64, len(RGB_AUX_EXTENDED_CHANNELS)))
                self.assertIn("aux_clipping_distance", [str(value) for value in payload["extended_channel_names"]])
                self.assertIn("aux_psf_edge_likelihood", [str(value) for value in payload["extended_channel_names"]])
                self.assertEqual(payload["boxes_xyxy"].shape[1], 4)
            self.assertEqual(rows[0]["extended_channels"], list(RGB_AUX_EXTENDED_CHANNELS))
            self.assertIn("extended_tensor_stats", rows[0])
            labels = json.loads(label_path.read_text())
            self.assertGreater(labels["box_count"], 0)
            self.assertEqual(summary["extended_channels"], list(RGB_AUX_EXTENDED_CHANNELS))
            self.assertIn("rgb_aux_extended_chw", summary["tensor_layouts"])
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "index.html").exists())

    def test_export_aux_dataset_minimal_writes_only_gate_tensors(self) -> None:
        samples = make_synthetic_evaluation_samples(count=1, width=64, height=48)
        with tempfile.TemporaryDirectory() as tmp:
            summary = export_aux_dataset(
                samples,
                tmp,
                include_extended=False,
                include_preview=False,
                compress=False,
            )
            root = Path(tmp)
            rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
            tensor_path = root / rows[0]["tensor_path"]
            with np.load(tensor_path) as payload:
                keys = set(payload.files)
                self.assertIn("rgb_aux_chw", keys)
                self.assertIn("rgb_aux_hwc", keys)
                self.assertNotIn("rgb_aux_extended_chw", keys)
                self.assertNotIn("perception_rgb_hwc", keys)
            self.assertEqual(summary["extended_channels"], [])
            self.assertEqual(summary["export_options"]["include_extended"], False)
            self.assertEqual(summary["export_options"]["include_preview"], False)
            self.assertEqual(summary["export_options"]["compress"], False)
            self.assertEqual(summary["tensor_layouts"], ["rgb_aux_hwc", "rgb_aux_chw"])
            self.assertEqual(rows[0]["extended_channels"], [])
            self.assertNotIn("extended_tensor_stats", rows[0])

    def test_export_aux_dataset_can_store_edge_evidence_as_base_six_channels(self) -> None:
        samples = make_synthetic_evaluation_samples(count=1, width=64, height=48)
        with tempfile.TemporaryDirectory() as tmp:
            summary = export_aux_dataset(
                samples,
                tmp,
                channels=EDGE6_CHANNELS,
                include_extended=False,
                include_preview=False,
                compress=False,
            )
            root = Path(tmp)
            rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
            tensor_path = root / rows[0]["tensor_path"]
            with np.load(tensor_path) as payload:
                self.assertEqual(tuple(str(value) for value in payload["channel_names"]), EDGE6_CHANNELS)
                self.assertEqual(payload["rgb_aux_chw"].shape, (len(EDGE6_CHANNELS), 48, 64))
                self.assertNotIn("rgb_aux_extended_chw", payload.files)
            self.assertEqual(summary["channels"], list(EDGE6_CHANNELS))
            self.assertEqual(rows[0]["channels"], list(EDGE6_CHANNELS))
            self.assertEqual(rows[0]["tensor_stats"]["channels"], list(EDGE6_CHANNELS))

    def test_cli_no_preview_keeps_extended_tensors_without_preview_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = aux_export_main(
                    [
                        "--source",
                        "synthetic",
                        "--count",
                        "1",
                        "--width",
                        "16",
                        "--height",
                        "12",
                        "--no-preview",
                        "--no-compress",
                        "--output-dir",
                        tmp,
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["export_options"]["include_extended"], True)
            self.assertEqual(summary["export_options"]["include_preview"], False)
            self.assertEqual(summary["run_config"]["no_preview"], True)
            self.assertEqual(summary["run_config"]["include_preview"], False)
            self.assertIn("rgb_aux_extended_chw", summary["tensor_layouts"])
            self.assertNotIn("perception_rgb_hwc", summary["tensor_layouts"])
            root = Path(tmp)
            rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
            with np.load(root / rows[0]["tensor_path"]) as payload:
                keys = set(payload.files)
                self.assertIn("rgb_aux_extended_chw", keys)
                self.assertIn("rgb_aux_extended_hwc", keys)
                self.assertNotIn("perception_rgb_hwc", keys)
                self.assertNotIn("perception_aux_hwc", keys)

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

    def test_load_samples_for_pascalraw_dataset_forwards_manifest_options(self) -> None:
        manifest = Path("reports/unit_pascalraw_manifest.json")
        with mock.patch("perception_isp.pascalraw_loader.load_pascalraw_detection_samples", return_value=()) as loader:
            samples = _load_samples(
                source="pascalraw-dataset",
                dataset="data/raw_datasets/pascalraw",
                split="val",
                count=3,
                offset=5,
                width=64,
                height=48,
                cfa_pattern="auto",
                use_camerae2e=False,
                progress_interval=2,
                pascalraw_manifest=manifest,
                pascalraw_native_raw=False,
            )

        self.assertEqual(samples, ())
        args = loader.call_args.args
        kwargs = loader.call_args.kwargs
        self.assertEqual(args[0], "data/raw_datasets/pascalraw")
        self.assertEqual(args[1], manifest)
        self.assertEqual(kwargs["limit"], 3)
        self.assertEqual(kwargs["offset"], 5)
        self.assertEqual(kwargs["width"], 64)
        self.assertEqual(kwargs["height"], 48)
        self.assertEqual(kwargs["cfa_pattern"], "auto")
        self.assertEqual(kwargs["use_camerae2e"], False)
        self.assertEqual(kwargs["progress_interval"], 2)
        self.assertEqual(kwargs["progress_label"], "load:pascalraw-dataset:5+3")

    def test_load_samples_for_pascalraw_native_dataset_uses_native_loader(self) -> None:
        manifest = Path("reports/unit_pascalraw_native_manifest.json")
        with mock.patch("perception_isp.pascalraw_loader.load_pascalraw_native_detection_samples", return_value=()) as loader:
            samples = _load_samples(
                source="pascalraw-dataset",
                dataset="data/raw_datasets/pascalraw",
                split="val",
                count=4,
                offset=7,
                width=80,
                height=60,
                cfa_pattern="RGGB",
                use_camerae2e=True,
                progress_interval=2,
                pascalraw_manifest=manifest,
                pascalraw_native_raw=True,
            )

        self.assertEqual(samples, ())
        args = loader.call_args.args
        kwargs = loader.call_args.kwargs
        self.assertEqual(args[0], "data/raw_datasets/pascalraw")
        self.assertEqual(args[1], manifest)
        self.assertEqual(kwargs["limit"], 4)
        self.assertEqual(kwargs["offset"], 7)
        self.assertEqual(kwargs["width"], 80)
        self.assertEqual(kwargs["height"], 60)
        self.assertEqual(kwargs["progress_interval"], 2)
        self.assertEqual(kwargs["progress_label"], "load:pascalraw-dataset:native:7+4")

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_torch_dataset_and_stem_consume_exported_tensors(self) -> None:
        import torch

        samples = make_synthetic_evaluation_samples(count=1, width=64, height=48)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp)
            dataset = make_torch_dataset(Path(tmp) / "manifest.jsonl")
            extended_dataset = make_torch_dataset(Path(tmp) / "manifest.jsonl", tensor_key="rgb_aux_extended_chw")
            tensor, target = dataset[0]
            extended_tensor, extended_target = extended_dataset[0]
            self.assertEqual(tuple(tensor.shape), (6, 48, 64))
            self.assertEqual(tuple(extended_tensor.shape), (len(RGB_AUX_EXTENDED_CHANNELS), 48, 64))
            self.assertIn("boxes_normalized", target)
            self.assertIn("boxes_normalized", extended_target)
            stem = make_aux_early_fusion_stem(in_channels=6, out_channels=8)
            out = stem(tensor.unsqueeze(0))
            self.assertEqual(out.shape[1], 8)
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(channels_for_tensor_key("rgb_aux_extended_chw"), RGB_AUX_EXTENDED_CHANNELS)

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
    def test_extended_smoke_training_loop_runs_on_exported_tensors(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp)
            summary = train_smoke(
                manifest_path=Path(tmp) / "manifest.jsonl",
                tensor_key="rgb_aux_extended_chw",
                epochs=1,
                device_name="cpu",
                eval_fraction=0.34,
                output_dir=Path(tmp) / "train_extended",
            )
            self.assertEqual(summary["tensor_key"], "rgb_aux_extended_chw")
            self.assertEqual(summary["input_channels"], len(RGB_AUX_EXTENDED_CHANNELS))
            self.assertEqual(summary["channels"], list(RGB_AUX_EXTENDED_CHANNELS))
            detector = RGBAuxTorchSmokeDetector(summary["checkpoint"], confidence=0.0)
            self.assertEqual(detector.input_channels, len(RGB_AUX_EXTENDED_CHANNELS))
            result = compare_dataset(samples[:1], rgb_aux_detector=detector, include_images=True)
            self.assertIn("perception_rgb_aux_dnn", result["aggregate"])

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
                batch_size=2,
                grid_size=(4, 6),
                base_channels=8,
                channel_mode="aux_only",
                eval_fraction=0.34,
                include_labels=("car", "person"),
                estimate_samples=(3, 30),
                output_dir=Path(tmp) / "dense",
            )
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["batch_size"], 2)
            self.assertEqual(summary["include_labels"], ["car", "person"])
            self.assertEqual(summary["class_names"], ["car", "person"])
            self.assertEqual(summary["split_strategy"], "coverage")
            self.assertIn("train_indices", summary)
            self.assertIn("eval_indices", summary)
            self.assertIn("checkpoint_epoch", summary)
            self.assertIn("checkpoint_loss", summary)
            self.assertIn("checkpoint_loss_kind", summary)
            self.assertEqual(summary["box_encoding"], "cell_center_size")
            self.assertEqual(summary["positive_cell_radius"], 0)
            self.assertEqual(summary["small_object_weight"], 1.0)
            self.assertEqual(summary["small_object_area_threshold"], 32.0 * 32.0)
            self.assertEqual(summary["channel_mode"], "aux_only")
            self.assertEqual(summary["object_loss_mode"], "balanced_positive_negative")
            self.assertEqual(summary["negative_focal_gamma"], 2.0)
            self.assertEqual(summary["model_architecture"], "early_fusion")
            self.assertEqual(summary["channel_mask"], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
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
            self.assertEqual(direct["checkpoint_summary"]["channel_mode"], "aux_only")
            self.assertEqual(direct["checkpoint_summary"]["model_architecture"], "early_fusion")
            self.assertTrue(all(sample["gt_count"] == 2 for sample in direct["samples"]))
            self.assertEqual(direct["input_ablation"], "none")
            self.assertIn("aggregate", direct)
            self.assertTrue((Path(tmp) / "dense_eval" / "dense_eval_summary.json").exists())
            self.assertTrue((Path(tmp) / "dense_eval" / "index.html").exists())
            zero_aux = evaluate_dense_manifest(
                manifest_path=manifest,
                checkpoint_path=summary["checkpoint"],
                split="eval",
                confidence=0.0,
                label_agnostic=False,
                input_ablation="zero_aux",
                output_dir=Path(tmp) / "dense_eval_zero_aux",
            )
            self.assertEqual(zero_aux["input_ablation"], "zero_aux")
            self.assertTrue(all(sample["input_ablation"] == "zero_aux" for sample in zero_aux["samples"]))
            self.assertTrue((Path(tmp) / "dense_eval_zero_aux" / "dense_eval_summary.json").exists())
            subset = evaluate_dense_manifest(
                manifest_path=manifest,
                checkpoint_path=summary["checkpoint"],
                split="eval",
                indices=summary["eval_indices"][:1],
                confidence=0.0,
                label_agnostic=False,
                output_dir=Path(tmp) / "dense_eval_subset",
            )
            self.assertEqual(subset["sample_count"], 1)
            self.assertTrue(subset["indices_override"])
            self.assertEqual(subset["selected_indices"], summary["eval_indices"][:1])
            capped = evaluate_dense_manifest(
                manifest_path=manifest,
                checkpoint_path=summary["checkpoint"],
                split="eval",
                confidence=0.0,
                max_detections=2,
                nms_iou=0.40,
                label_agnostic=False,
                output_dir=Path(tmp) / "dense_eval_capped",
            )
            self.assertEqual(capped["detector_config"]["max_detections"], 2)
            self.assertEqual(capped["detector_config"]["nms_iou"], 0.40)
            self.assertLessEqual(capped["aggregate"]["det_count_mean"], 2.0)
            self.assertGreaterEqual(direct["aggregate"]["det_count_mean"], capped["aggregate"]["det_count_mean"])

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_dense_hash_split_is_not_sequential_tail(self) -> None:
        samples = make_synthetic_evaluation_samples(count=8, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp, include_extended=False, include_preview=False, compress=False)
            summary = train_dense(
                manifest_path=Path(tmp) / "manifest.jsonl",
                epochs=1,
                device_name="cpu",
                grid_size=(4, 6),
                base_channels=8,
                eval_fraction=0.5,
                split_strategy="hash",
                include_labels=("car", "person"),
                object_loss_mode="negative_focal",
                negative_focal_gamma=1.5,
                output_dir=Path(tmp) / "dense_hash",
            )
            self.assertEqual(summary["split_strategy"], "hash")
            self.assertEqual(summary["object_loss_mode"], "negative_focal")
            self.assertEqual(summary["negative_focal_gamma"], 1.5)
            self.assertEqual(summary["train_sample_count"], 4)
            self.assertEqual(summary["eval_sample_count"], 4)
            self.assertNotEqual(summary["train_indices"], [0, 1, 2, 3])
            self.assertEqual(summary["missing_eval_class_names"], [])

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_dense_training_can_warm_start_from_rgb_only_checkpoint(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp, include_preview=False, compress=False)
            manifest = Path(tmp) / "manifest.jsonl"
            rgb_only = train_dense(
                manifest_path=manifest,
                tensor_key="rgb_aux_extended_chw",
                epochs=1,
                device_name="cpu",
                grid_size=(4, 6),
                base_channels=8,
                channel_mode="rgb_only",
                eval_fraction=0.34,
                include_labels=("car", "person"),
                seed=7,
                output_dir=Path(tmp) / "rgb_only",
            )
            warm = train_dense(
                manifest_path=manifest,
                tensor_key="rgb_aux_extended_chw",
                epochs=1,
                device_name="cpu",
                grid_size=(4, 6),
                base_channels=8,
                channel_mode="rgb_aux",
                eval_fraction=0.34,
                include_labels=("car", "person"),
                seed=11,
                initial_checkpoint=rgb_only["checkpoint"],
                zero_aux_input_weights=True,
                save_epoch_checkpoints=True,
                output_dir=Path(tmp) / "warm",
            )
            self.assertEqual(warm["initial_checkpoint"], rgb_only["checkpoint"])
            self.assertEqual(warm["seed"], 11)
            self.assertTrue(warm["seed_info"]["set"])
            self.assertTrue(warm["initialization"]["loaded"])
            self.assertEqual(warm["initialization"]["checkpoint_channel_mode"], "rgb_only")
            zero_result = warm["initialization"]["zero_aux_input_weight_result"]
            self.assertEqual(zero_result["status"], "zeroed")
            self.assertEqual(zero_result["input_channels"], len(RGB_AUX_EXTENDED_CHANNELS))
            self.assertEqual(zero_result["aux_start_channel"], 3)
            self.assertEqual(zero_result["abs_sum_after"], 0.0)
            self.assertTrue(warm["save_epoch_checkpoints"])
            self.assertEqual(len(warm["epoch_checkpoints"]), 1)
            self.assertTrue(Path(warm["epoch_checkpoints"][0]["checkpoint"]).exists())
            self.assertTrue((Path(tmp) / "warm" / "rgb_aux_dense_detector.pt").exists())

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_dense_late_fusion_detector_trains_and_evaluates(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp, include_extended=False, include_preview=False, compress=False)
            summary = train_dense(
                manifest_path=Path(tmp) / "manifest.jsonl",
                epochs=1,
                device_name="cpu",
                grid_size=(4, 6),
                base_channels=8,
                model_architecture="late_fusion",
                channel_mode="rgb_aux",
                eval_fraction=0.34,
                include_labels=("car", "person"),
                output_dir=Path(tmp) / "dense_late",
            )
            self.assertEqual(summary["model_architecture"], "late_fusion")
            detector = rgb_aux_detector_from_checkpoint(summary["checkpoint"], confidence=0.0)
            self.assertIsInstance(detector, RGBAuxTorchDenseDetector)
            self.assertEqual(detector.model_architecture, "late_fusion")
            direct = evaluate_dense_manifest(
                manifest_path=Path(tmp) / "manifest.jsonl",
                checkpoint_path=summary["checkpoint"],
                split="eval",
                confidence=0.0,
                label_agnostic=False,
                output_dir=Path(tmp) / "dense_late_eval",
            )
            self.assertEqual(direct["checkpoint_summary"]["model_architecture"], "late_fusion")
            self.assertIn("aggregate", direct)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_dense_multiscale_detector_trains_and_evaluates(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp, include_extended=False, include_preview=False, compress=False)
            summary = train_dense(
                manifest_path=Path(tmp) / "manifest.jsonl",
                epochs=1,
                device_name="cpu",
                grid_size=(4, 6),
                base_channels=8,
                model_architecture="early_fusion_multiscale",
                channel_mode="rgb_aux",
                eval_fraction=0.34,
                include_labels=("car", "person"),
                output_dir=Path(tmp) / "dense_multiscale",
            )
            self.assertEqual(summary["model_architecture"], "early_fusion_multiscale")
            detector = rgb_aux_detector_from_checkpoint(summary["checkpoint"], confidence=0.0)
            self.assertIsInstance(detector, RGBAuxTorchDenseDetector)
            self.assertEqual(detector.model_architecture, "early_fusion_multiscale")
            direct = evaluate_dense_manifest(
                manifest_path=Path(tmp) / "manifest.jsonl",
                checkpoint_path=summary["checkpoint"],
                split="eval",
                confidence=0.0,
                label_agnostic=False,
                output_dir=Path(tmp) / "dense_multiscale_eval",
            )
            self.assertEqual(direct["checkpoint_summary"]["model_architecture"], "early_fusion_multiscale")
            self.assertIn("aggregate", direct)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_extended_dense_detector_trains_and_evaluates(self) -> None:
        samples = make_synthetic_evaluation_samples(count=3, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp)
            manifest = Path(tmp) / "manifest.jsonl"
            summary = train_dense(
                manifest_path=manifest,
                tensor_key="rgb_aux_extended_chw",
                epochs=1,
                device_name="cpu",
                grid_size=(4, 6),
                base_channels=8,
                channel_mode="aux_only",
                eval_fraction=0.34,
                include_labels=("car", "person"),
                output_dir=Path(tmp) / "dense_extended",
            )
            self.assertEqual(summary["tensor_key"], "rgb_aux_extended_chw")
            self.assertEqual(summary["input_channels"], len(RGB_AUX_EXTENDED_CHANNELS))
            self.assertEqual(summary["channels"], list(RGB_AUX_EXTENDED_CHANNELS))
            self.assertEqual(len(summary["channel_mask"]), len(RGB_AUX_EXTENDED_CHANNELS))
            self.assertEqual(summary["channel_mask"][:3], [0.0, 0.0, 0.0])
            self.assertTrue(all(value == 1.0 for value in summary["channel_mask"][3:]))
            detector = rgb_aux_detector_from_checkpoint(summary["checkpoint"], confidence=0.0)
            self.assertIsInstance(detector, RGBAuxTorchDenseDetector)
            self.assertEqual(detector.input_channels, len(RGB_AUX_EXTENDED_CHANNELS))
            result = compare_dataset(samples[:1], rgb_aux_detector=detector, label_agnostic=False, include_images=True)
            self.assertIn("perception_rgb_aux_dnn", result["aggregate"])
            direct = evaluate_dense_manifest(
                manifest_path=manifest,
                checkpoint_path=summary["checkpoint"],
                split="eval",
                confidence=0.0,
                label_agnostic=False,
                output_dir=Path(tmp) / "dense_extended_eval",
            )
            self.assertEqual(direct["checkpoint_summary"]["tensor_key"], "rgb_aux_extended_chw")
            self.assertEqual(direct["checkpoint_summary"]["input_channels"], len(RGB_AUX_EXTENDED_CHANNELS))
            self.assertIn("aggregate", direct)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_dense_xyxy_radius_training_writes_matching_checkpoint(self) -> None:
        samples = make_synthetic_evaluation_samples(count=4, width=48, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            export_aux_dataset(samples, tmp, include_extended=False, include_preview=False, compress=False)
            summary = train_dense(
                manifest_path=Path(tmp) / "manifest.jsonl",
                epochs=1,
                device_name="cpu",
                batch_size=2,
                grid_size=(4, 6),
                base_channels=8,
                box_encoding="xyxy",
                positive_cell_radius=1,
                eval_fraction=0.25,
                include_labels=("car", "person"),
                output_dir=Path(tmp) / "dense_xyxy",
            )
            self.assertEqual(summary["box_encoding"], "xyxy")
            self.assertEqual(summary["positive_cell_radius"], 1)
            self.assertEqual(summary["small_object_weight"], 1.0)
            detector = rgb_aux_detector_from_checkpoint(summary["checkpoint"], confidence=0.0)
            self.assertEqual(detector.box_encoding, "xyxy")
            direct = evaluate_dense_manifest(
                manifest_path=Path(tmp) / "manifest.jsonl",
                checkpoint_path=summary["checkpoint"],
                split="eval",
                confidence=0.0,
                label_agnostic=False,
                output_dir=Path(tmp) / "dense_xyxy_eval",
            )
            self.assertEqual(direct["checkpoint_summary"]["tensor_key"], "rgb_aux_chw")
            self.assertIn("aggregate", direct)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_dense_targets_weight_small_objects(self) -> None:
        import torch

        target = {
            "boxes": torch.tensor(
                [
                    [2.0, 2.0, 12.0, 12.0],
                    [32.0, 32.0, 96.0, 96.0],
                ],
                dtype=torch.float32,
            ),
            "boxes_normalized": torch.tensor(
                [
                    [0.05, 0.05, 0.20, 0.20],
                    [0.60, 0.60, 0.90, 0.90],
                ],
                dtype=torch.float32,
            ),
            "labels_text": ("car", "person"),
        }
        object_target, _, class_target, positive_weight = _dense_targets(
            target,
            {"car": 0, "person": 1},
            (4, 4),
            torch.device("cpu"),
            torch,
            box_encoding="xyxy",
            positive_cell_radius=0,
            small_object_weight=3.0,
            small_object_area_threshold=32.0 * 32.0,
        )
        positives = object_target[0] > 0.5
        self.assertEqual(int(torch.sum(positives).item()), 2)
        self.assertEqual(float(torch.max(positive_weight[positives]).item()), 3.0)
        self.assertEqual(float(torch.min(positive_weight[positives]).item()), 1.0)
        self.assertEqual(set(int(value.item()) for value in class_target[positives]), {0, 1})

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_weighted_mean_uses_weights_as_loss_multipliers(self) -> None:
        import torch

        values = torch.tensor([2.0, 4.0], dtype=torch.float32)
        weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
        self.assertAlmostEqual(float(_weighted_mean(values, weights, torch).item()), 7.0)


if __name__ == "__main__":
    unittest.main()
