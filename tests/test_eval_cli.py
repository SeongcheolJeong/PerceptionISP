from __future__ import annotations

import unittest

import numpy as np

from perception_isp.eval_cli import (
    apply_psf_sigma_to_samples,
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


if __name__ == "__main__":
    unittest.main()
