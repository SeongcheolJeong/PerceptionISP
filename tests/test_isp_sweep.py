from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from perception_isp.isp_sweep import build_sweep_summary, config_id, iter_sweep_configs, parse_csv, parse_float_csv, summarize_run
from perception_isp.types import PerceptionISPConfig


class ISPSweepTest(unittest.TestCase):
    def test_parse_sweep_values(self) -> None:
        self.assertEqual(parse_csv("log, srgb"), ("log", "srgb"))
        self.assertEqual(parse_float_csv("0.0, 0.18"), (0.0, 0.18))

    def test_iter_sweep_configs_expands_grid_and_names_configs(self) -> None:
        configs = iter_sweep_configs(
            tone_mappings=("log", "srgb"),
            denoise_strengths=(0.0, 0.18),
            demosaic_methods=("edge_aware",),
            demosaic_artifact_suppressions=(0.20,),
        )
        self.assertEqual(len(configs), 4)
        self.assertIn("tone-log", config_id(configs[0]))
        self.assertIn("denoise-0.00", config_id(configs[0]))

    def test_summarize_run_computes_deltas_vs_human(self) -> None:
        result = {
            "run_config": {
                "run_index": 1,
                "run_id": "tone-log",
                "perception_config": {"tone_mapping": "log"},
                "human_baseline_config": {"tone_mapping": "log"},
            },
            "aggregate": {
                "human_rgb": {"recall@0.50_mean": 0.40, "small_recall@0.50_mean": 0.20},
                "perception_rgb": {"recall@0.50_mean": 0.45, "small_recall@0.50_mean": 0.18},
            },
        }
        summary = summarize_run(result, Path("001_tone-log/index.html"))
        self.assertAlmostEqual(summary["delta_vs_human"]["perception_rgb"]["recall@0.50_mean"], 0.05)
        self.assertAlmostEqual(summary["delta_vs_human"]["perception_rgb"]["small_recall@0.50_mean"], -0.02)

    def test_build_sweep_summary_selects_best_delta(self) -> None:
        args = Namespace(
            source="yolo-dataset",
            dataset="data/kitti/data.yaml",
            split="val",
            count=2,
            offset=0,
            width=640,
            height=192,
            cfa="auto",
            no_camerae2e=False,
            load_progress_interval=0,
            raw_cache_dir=None,
            label_aware=True,
        )
        runs = [
            {
                "run_id": "a",
                "report": "a/index.html",
                "perception_config": {"tone_mapping": "log"},
                "delta_vs_human": {"perception_rgb": {"recall@0.50_mean": -0.01, "small_recall@0.50_mean": 0.0}},
            },
            {
                "run_id": "b",
                "report": "b/index.html",
                "perception_config": {"tone_mapping": "srgb"},
                "delta_vs_human": {"perception_rgb": {"recall@0.50_mean": 0.02, "small_recall@0.50_mean": 0.01}},
            },
        ]
        summary = build_sweep_summary(args, runs, {"pedestrian": "person"}, PerceptionISPConfig())
        self.assertEqual(summary["best"]["perception_rgb_by_delta_recall@0.50"]["run_id"], "b")
        self.assertEqual(summary["ground_truth_label_map"]["pedestrian"], "person")


if __name__ == "__main__":
    unittest.main()
