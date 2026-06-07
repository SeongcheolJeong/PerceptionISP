from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.aodraw_loader import load_aodraw_detection_samples, load_aodraw_manifest_row


class AODRawLoaderTest(unittest.TestCase):
    def test_load_manifest_row_builds_evaluation_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _write_sample(root, "00000001")

            sample = load_aodraw_manifest_row(
                root,
                row,
                target_width=4,
                target_height=4,
                cfa_pattern="GRBG",
                black_level=16.0,
                white_level=1023.0,
                require_srgb=True,
            )

            self.assertEqual(sample.sample_id, "00000001")
            self.assertEqual(sample.source, "aodraw_native_raw_manifest")
            self.assertEqual(sample.raw.data.shape, (4, 4))
            self.assertEqual(sample.raw.metadata.cfa_pattern, "GRBG")
            self.assertEqual(sample.raw.calibration.cfa_pattern, "GRBG")
            self.assertEqual(sample.raw.calibration.black_level, 16.0)
            self.assertEqual(sample.raw.calibration.white_level, 1023.0)
            self.assertEqual(sample.raw.provenance["raw_resize_mode"], "nearest_cfa_preserve_parity")
            self.assertFalse(sample.raw.provenance["native_resolution_matches_target"])
            self.assertTrue(sample.raw.provenance["true_sensor_cfa_mosaic"])
            self.assertEqual(sample.reference_rgb.shape, (4, 4, 3))
            self.assertEqual(len(sample.ground_truth), 1)
            self.assertEqual(sample.ground_truth[0].xyxy, (1.0, 1.0, 3.0, 3.0))
            self.assertEqual(sample.ground_truth[0].label, "person")

    def test_load_samples_accepts_summary_payload_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [_write_sample(root, "00000001"), _write_sample(root, "00000002")]
            manifest_path = root / "subset_summary.json"
            manifest_path.write_text(json.dumps({"manifest": rows}))

            samples = load_aodraw_detection_samples(root, manifest_path, limit=1, offset=1, width=8, height=8)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "00000002")
            self.assertEqual(samples[0].raw.provenance["raw_resize_mode"], "native")
            self.assertTrue(samples[0].raw.provenance["native_resolution_matches_target"])

    def test_missing_raw_file_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = _manifest_row("missing")
            with self.assertRaises(FileNotFoundError):
                load_aodraw_manifest_row(Path(tmp), row)

    def test_require_srgb_fails_when_reference_image_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _write_sample(root, "00000001", write_srgb=False)
            with self.assertRaises(FileNotFoundError):
                load_aodraw_manifest_row(root, row, require_srgb=True)


def _write_sample(root: Path, stem: str, *, write_srgb: bool = True) -> dict:
    raw_path = root / "images_downsampled_raw" / f"{stem}.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
    np.save(raw_path, raw)
    if write_srgb:
        srgb_path = root / "images_downsampled_srgb" / f"{stem}.JPG"
        srgb_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 0] = 100
        image[:, :, 1] = 120
        image[:, :, 2] = 140
        Image.fromarray(image).save(srgb_path)
    return _manifest_row(stem)


def _manifest_row(stem: str) -> dict:
    return {
        "image_id": int(stem) if stem.isdigit() else -1,
        "file_name": f"{stem}.JPG",
        "selection_condition": "low_light",
        "tags": ["low_light"],
        "width": 8,
        "height": 8,
        "box_count": 1,
        "expected_raw_relative_path": f"images_downsampled_raw/{stem}.npy",
        "expected_srgb_relative_path": f"images_downsampled_srgb/{stem}.JPG",
        "boxes": [{"xyxy": [2.0, 2.0, 6.0, 6.0], "label": "person", "area": 16.0}],
    }


if __name__ == "__main__":
    unittest.main()
