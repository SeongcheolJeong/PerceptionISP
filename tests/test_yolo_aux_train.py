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

        from perception_isp.training.yolo_aux_train import zero_aux_input_weights_for_model

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

        from perception_isp.training.yolo_aux_train import zero_aux_input_weights_for_model

        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU())
        result = zero_aux_input_weights_for_model(model, aux_start_channel=3)

        self.assertEqual(result["status"], "not_found")

    def test_gated_stem_replacement_is_idempotent_for_existing_gated_stem(self) -> None:
        import torch.nn as nn

        from perception_isp.training.yolo_aux_train import RgbAuxGatedStem, replace_first_stem_with_gated_aux_stem

        class FakeStem(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(8)
                self.act = nn.SiLU(inplace=True)

        stem = RgbAuxGatedStem(
            FakeStem(),
            aux_start_channel=3,
            aux_channels=3,
            init_mode="mean_rgb",
            adapter_checkpoint=None,
            adapter_scale=1.0,
            gate_init=-2.0,
            freeze_rgb_branch=False,
        )
        model = types.SimpleNamespace(model=nn.ModuleList([stem, nn.Conv2d(8, 16, kernel_size=3)]))

        result = replace_first_stem_with_gated_aux_stem(model, init_mode="adapter", adapter_checkpoint=None)

        self.assertEqual(result["status"], "already_initialized")
        self.assertIs(model.model[0], stem)
        self.assertTrue(result["matches_requested"])

    def test_gated_norm_stem_adds_aux_normalization(self) -> None:
        import torch
        import torch.nn as nn

        from perception_isp.training.yolo_aux_train import replace_first_stem_with_gated_aux_stem

        class FakeStem(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(8)
                self.act = nn.SiLU(inplace=True)

        model = types.SimpleNamespace(model=nn.ModuleList([FakeStem()]))

        result = replace_first_stem_with_gated_aux_stem(model, init_mode="mean_rgb", mode="gated_norm_sum")

        self.assertEqual(result["status"], "initialized")
        self.assertEqual(result["mode"], "gated_norm_sum")
        self.assertTrue(result["aux_norm"])
        self.assertTrue(getattr(model.model[0], "aux_norm_enabled"))
        self.assertIsInstance(model.model[0].aux_bn, nn.BatchNorm2d)
        output = model.model[0](torch.randn(2, 6, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 8, 16, 16))

    def test_gated_stem_forward_supports_old_checkpoints_without_aux_bn(self) -> None:
        import torch
        import torch.nn as nn

        from perception_isp.training.yolo_aux_train import RgbAuxGatedStem

        class FakeStem(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(8)
                self.act = nn.SiLU(inplace=True)

        stem = RgbAuxGatedStem(
            FakeStem(),
            aux_start_channel=3,
            aux_channels=3,
            init_mode="mean_rgb",
            adapter_checkpoint=None,
            adapter_scale=1.0,
            gate_init=-2.0,
            freeze_rgb_branch=False,
        )
        delattr(stem, "aux_bn")

        output = stem(torch.randn(2, 6, 32, 32))

        self.assertEqual(tuple(output.shape), (2, 8, 16, 16))

    def test_gated_stem_replacement_expands_rgb_first_stem(self) -> None:
        import torch
        import torch.nn as nn

        from perception_isp.training.yolo_aux_train import replace_first_stem_with_gated_aux_stem

        class FakeRgbStem(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(8)
                self.act = nn.SiLU(inplace=True)

        second = nn.Conv2d(16, 16, kernel_size=3)
        model = types.SimpleNamespace(model=nn.ModuleList([FakeRgbStem(), second]))

        result = replace_first_stem_with_gated_aux_stem(
            model,
            init_mode="mean_rgb",
            freeze_rgb_branch=True,
        )

        self.assertEqual(result["status"], "initialized")
        self.assertTrue(result["expanded_from_rgb_stem"])
        self.assertIs(model.model[1], second)
        self.assertTrue(hasattr(model.model[0], "aux_conv"))
        self.assertFalse(model.model[0].rgb_conv.weight.requires_grad)
        output = model.model[0](torch.randn(2, 6, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 8, 16, 16))

    def test_restore_first_stem_from_gated_source_replaces_first_stem(self) -> None:
        import torch.nn as nn

        from perception_isp.training.yolo_aux_train import RgbAuxGatedStem, restore_first_stem_from_gated_source

        class FakeStem(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(8)
                self.act = nn.SiLU(inplace=True)

        source = RgbAuxGatedStem(
            FakeStem(),
            aux_start_channel=3,
            aux_channels=3,
            init_mode="mean_rgb",
            adapter_checkpoint=None,
            adapter_scale=1.0,
            gate_init=-2.0,
            freeze_rgb_branch=False,
        )
        model = types.SimpleNamespace(model=nn.ModuleList([FakeStem()]))

        result = restore_first_stem_from_gated_source(model, source_stem=source)

        self.assertEqual(result["status"], "restored_from_checkpoint")
        self.assertIsNot(model.model[0], source)
        self.assertTrue(hasattr(model.model[0], "aux_conv"))
        self.assertTrue(hasattr(model.model[0], "gate_logit"))

    def test_train_yolo_aux_passes_seed_and_records_summary(self) -> None:
        from perception_isp.training.yolo_aux_train import train_yolo_aux

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
                    optimizer="AdamW",
                    lr0=0.0005,
                    lrf=0.05,
                    warmup_epochs=1.0,
                    fraction=0.25,
                    disable_augment=False,
                )
        finally:
            if previous is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = previous

        self.assertEqual(calls[0]["seed"], 7)
        self.assertEqual(calls[0]["optimizer"], "AdamW")
        self.assertEqual(calls[0]["lr0"], 0.0005)
        self.assertEqual(calls[0]["lrf"], 0.05)
        self.assertEqual(calls[0]["warmup_epochs"], 1.0)
        self.assertEqual(calls[0]["fraction"], 0.25)
        self.assertEqual(summary["seed"], 7)
        self.assertEqual(summary["train_overrides"]["optimizer"], "AdamW")
        self.assertEqual(summary["train_overrides"]["lr0"], 0.0005)
        self.assertEqual(summary["train_overrides"]["fraction"], 0.25)
        self.assertEqual(summary["results_dict"]["metrics/mAP50(B)"], 0.5)

    def test_train_yolo_aux_installs_aux_stem_before_optimizer_path(self) -> None:
        import torch.nn as nn

        from perception_isp.training.yolo_aux_train import train_yolo_aux

        calls = []

        class FakeRgbStem(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(8)
                self.act = nn.SiLU(inplace=True)

        class FakeBaseTrainer:
            def __init__(self, *_args, **_kwargs) -> None:
                self.model = None

            def setup_model(self):
                self.model = types.SimpleNamespace(model=nn.ModuleList([FakeRgbStem()]))
                return {"fake": "ckpt"}

        class FakeYOLO:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name
                self.model = types.SimpleNamespace(model=nn.ModuleList([FakeRgbStem()]))
                self.trainer = types.SimpleNamespace(save_dir="")
                self.callbacks = []

            def _smart_load(self, name: str):
                self.smart_load_name = name
                return FakeBaseTrainer

            def add_callback(self, event, callback) -> None:
                self.callbacks.append((event, callback))

            def train(self, trainer=None, **kwargs):
                self.trainer = types.SimpleNamespace(save_dir=Path(str(kwargs["project"])) / str(kwargs["name"]))
                calls.append({"trainer": trainer, "kwargs": dict(kwargs)})
                self.created_trainer = trainer()
                self.created_trainer.setup_model()
                for parameter in self.created_trainer.model.model[0].rgb_conv.parameters():
                    parameter.requires_grad_(True)
                for event, callback in self.callbacks:
                    if event == "on_pretrain_routine_end":
                        callback(types.SimpleNamespace(model=self.created_trainer.model, ema=None))
                return types.SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.6})

        previous = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=FakeYOLO)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                summary = train_yolo_aux(
                    data="data.yaml",
                    model_name="fake.pt",
                    epochs=1,
                    project=temp_dir,
                    name="aux_stem",
                    seed=3,
                    aux_stem_mode="gated_sum",
                    aux_stem_init="mean_rgb",
                    aux_stem_gate_init=-1.0,
                    aux_stem_freeze_rgb_branch=True,
                )
        finally:
            if previous is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = previous

        self.assertIsNotNone(calls[0]["trainer"])
        self.assertEqual(summary["aux_stem"]["status"], "initialized")
        self.assertEqual(summary["aux_stem"]["setup_phase"], "setup_model_before_optimizer")
        self.assertTrue(summary["aux_stem"]["expanded_from_rgb_stem"])
        self.assertEqual(summary["aux_stem"]["gate_init_logit"], -1.0)
        self.assertEqual(summary["aux_stem"]["freeze_enforced_after_ultralytics_freeze"]["status"], "frozen")
        self.assertFalse(any(summary["aux_stem"]["freeze_enforced_after_ultralytics_freeze"]["requires_grad_after"]))
        self.assertEqual(summary["results_dict"]["metrics/mAP50(B)"], 0.6)


if __name__ == "__main__":
    unittest.main()
