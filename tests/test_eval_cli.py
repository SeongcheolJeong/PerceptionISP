from __future__ import annotations

import unittest

import numpy as np

from perception_isp.eval_cli import parse_label_map, remap_sample_labels
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


if __name__ == "__main__":
    unittest.main()
