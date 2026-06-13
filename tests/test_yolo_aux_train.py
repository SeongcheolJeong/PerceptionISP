from __future__ import annotations

import importlib.util
import unittest


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


if __name__ == "__main__":
    unittest.main()
