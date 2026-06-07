from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.eval_cli import (
    apply_psf_sigma_to_samples,
    main as eval_cli_main,
    parse_label_map,
    raw_height_width,
    remap_sample_labels,
)
from perception_isp.eval_types import BoundingBox, EvaluationSample
from perception_isp.types import RawFrame, SensorMetadata


class EvalCliHelpersTest(unittest.TestCase):
    def test_parse_kitti_coco_label_map_preset(self) -> None:
        mapping = parse_label_map("kitti-coco")
        self.assertEqual(mapping["pedestrian"], "person")
        self.assertEqual(mapping["van"], "car")
        self.assertEqual(mapping["cyclist"], "bicycle")
        self.assertEqual(mapping["Person_sitting"], "person")

    def test_parse_custom_label_map(self) -> None:
        self.assertEqual(parse_label_map("pedestrian=person,van=car"), {"pedestrian": "person", "van": "car"})

    def test_remap_sample_labels_updates_boxes_and_metadata(self) -> None:
        sample = EvaluationSample(
            sample_id="sample",
            raw=RawFrame(data=np.zeros((1, 1), dtype=float), metadata=SensorMetadata(cfa_pattern="RGGB")),
            ground_truth=(BoundingBox((0, 0, 1, 1), label="pedestrian"), BoundingBox((1, 1, 2, 2), label="car")),
        )

        remapped = remap_sample_labels((sample,), {"pedestrian": "person"})

        self.assertEqual(remapped[0].ground_truth[0].label, "person")
        self.assertEqual(remapped[0].ground_truth[1].label, "car")
        self.assertEqual(remapped[0].metadata["ground_truth_label_remapped_count"], 1)

    def test_apply_psf_sigma_to_samples_updates_raw_calibration_and_metadata(self) -> None:
        raw = RawFrame(
            data=np.zeros((6, 8), dtype=float),
            metadata=SensorMetadata(cfa_pattern="GRBG", calibration_id="cal", lens_profile_id="lens"),
            provenance={"source": "unit"},
        )
        sample = EvaluationSample(sample_id="sample", raw=raw, ground_truth=(), metadata={"cfa_pattern": "GRBG"})

        conditioned = apply_psf_sigma_to_samples((sample,), 1.25)

        self.assertEqual(len(conditioned), 1)
        self.assertEqual(conditioned[0].metadata["psf_sigma"], 1.25)
        self.assertEqual(conditioned[0].raw.metadata.calibration_id, "cal_psf_1.25")
        self.assertEqual(conditioned[0].raw.metadata.lens_profile_id, "lens_psf_1.25")
        self.assertEqual(conditioned[0].raw.provenance["source"], "unit")
        self.assertEqual(conditioned[0].raw.provenance["eval_psf_sigma"], 1.25)
        self.assertEqual(conditioned[0].raw.calibration.psf_sigma_map.shape, (6, 8))
        self.assertAlmostEqual(float(np.mean(conditioned[0].raw.calibration.psf_sigma_map)), 1.25)

    def test_apply_psf_sigma_to_samples_clamps_negative_values(self) -> None:
        sample = EvaluationSample(
            sample_id="sample",
            raw=RawFrame(data=np.zeros((1, 4, 5), dtype=float), metadata=SensorMetadata(cfa_pattern="RGGB")),
            ground_truth=(),
        )

        conditioned = apply_psf_sigma_to_samples((sample,), -0.5)

        self.assertEqual(conditioned[0].raw.calibration.psf_sigma_map.shape, (4, 5))
        self.assertAlmostEqual(float(np.max(conditioned[0].raw.calibration.psf_sigma_map)), 0.0)
        self.assertEqual(conditioned[0].metadata["psf_sigma"], 0.0)

    def test_raw_height_width_accepts_hwc_and_ehw_raw_shapes(self) -> None:
        self.assertEqual(raw_height_width(np.zeros((3, 7, 9), dtype=float)), (7, 9))
        self.assertEqual(raw_height_width(np.zeros((7, 9, 3), dtype=float)), (7, 9))

    def test_cli_can_run_aodraw_dataset_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_aodraw_cli_sample(root)
            manifest_path = root / "manifest.json"
            output_dir = root / "report"

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = eval_cli_main(
                    [
                        "--source",
                        "aodraw-dataset",
                        "--dataset",
                        str(root),
                        "--aodraw-manifest",
                        str(manifest_path),
                        "--count",
                        "1",
                        "--width",
                        "8",
                        "--height",
                        "8",
                        "--rgb-detector",
                        "numpy",
                        "--no-visuals",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "comparison_summary.json").exists())
            self.assertEqual(printed["run_config"]["source"], "aodraw-dataset")
            self.assertFalse(printed["run_config"]["use_camerae2e"])
            self.assertEqual(printed["run_config"]["aodraw_cfa"], "RGGB")


def _write_aodraw_cli_sample(root: Path) -> None:
    raw_path = root / "images_downsampled_raw" / "00000001.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic = np.zeros((8, 8), dtype=np.uint16)
    mosaic[2:6, 2:6] = 900
    np.save(raw_path, mosaic)
    srgb_path = root / "images_downsampled_srgb" / "00000001.JPG"
    srgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[2:6, 2:6, :] = 180
    Image.fromarray(rgb).save(srgb_path)
    manifest = [
        {
            "image_id": 1,
            "file_name": "00000001.JPG",
            "selection_condition": "low_light",
            "tags": ["low_light"],
            "width": 8,
            "height": 8,
            "box_count": 1,
            "expected_raw_relative_path": "images_downsampled_raw/00000001.npy",
            "expected_srgb_relative_path": "images_downsampled_srgb/00000001.JPG",
            "boxes": [{"xyxy": [2.0, 2.0, 6.0, 6.0], "label": "person", "area": 16.0}],
        }
    ]
    (root / "manifest.json").write_text(json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
