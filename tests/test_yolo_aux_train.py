from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class YoloAuxTrainTest(unittest.TestCase):
    def test_zero_aux_input_weights_for_model(self) -> None:
        import torch.nn as nn

        from perception_isp.yolo_aux_train import zero_aux_input_weights_for_model

        model = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.ReLU())
        before_rgb = model[0].weight[:, :3, :, :].detach().clone()
        result = zero_aux_input_weights_for_model(model, aux_start_channel=3)

        self.assertEqual(result["status"], "zeroed")
        self.assertEqual(result["input_channels"], 16)
        self.assertGreater(result["abs_sum_before"], 0.0)
        self.assertEqual(result["abs_sum_after"], 0.0)
        self.assertTrue((model[0].weight[:, 3:, :, :] == 0).all())
        self.assertTrue((model[0].weight[:, :3, :, :] == before_rgb).all())

    def test_zero_aux_input_weights_reports_not_found_for_rgb_model(self) -> None:
        import torch.nn as nn

        from perception_isp.yolo_aux_train import zero_aux_input_weights_for_model

        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU())
        result = zero_aux_input_weights_for_model(model, aux_start_channel=3)

        self.assertEqual(result["status"], "not_found")

    def test_train_yolo_aux_passes_seed_and_records_summary(self) -> None:
        from perception_isp.yolo_aux_train import train_yolo_aux

        calls = []

        class FakeYOLO:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name
                self.trainer = types.SimpleNamespace(save_dir="")

            def add_callback(self, *_args, **_kwargs) -> None:
                raise AssertionError("callback should not be registered without zero-aux warm start")

            def train(self, **kwargs):
                calls.append(dict(kwargs))
                self.trainer.save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                return types.SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.5})

        previous = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=FakeYOLO)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                summary = train_yolo_aux(
                    data="data.yaml",
                    model_name="fake.pt",
                    epochs=1,
                    project=temp_dir,
                    name="seed7",
                    seed=7,
                    disable_augment=False,
                )
        finally:
            if previous is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = previous

        self.assertEqual(calls[0]["seed"], 7)
        self.assertEqual(summary["seed"], 7)
        self.assertEqual(summary["results_dict"]["metrics/mAP50(B)"], 0.5)


if __name__ == "__main__":
    unittest.main()
