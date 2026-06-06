from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.aux_training_rollup import build_training_rollup, main as rollup_main, write_training_rollup


class AuxTrainingRollupTest(unittest.TestCase):
    def test_rollup_combines_training_export_and_eval_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "export"
            train_dir = root / "train"
            eval_dir = root / "eval"
            export_dir.mkdir()
            train_dir.mkdir()
            eval_dir.mkdir()
            (eval_dir / "index.html").write_text("<html></html>")
            (export_dir / "summary.json").write_text(
                json.dumps({"sample_count": 4, "elapsed_seconds": 2.0, "samples_per_second": 2.0}) + "\n"
            )
            (train_dir / "train_dense_summary.json").write_text(
                json.dumps(
                    {
                        "sample_count": 4,
                        "train_sample_count": 3,
                        "eval_sample_count": 1,
                        "epochs": 2,
                        "device": "cpu",
                        "channel_mode": "rgb_aux",
                        "elapsed_seconds": 1.0,
                        "sample_epochs_per_second": 6.0,
                        "final_eval_loss": 1.25,
                        "time_estimates": [{"samples": 40, "epochs": 2, "estimated_minutes": 0.25, "estimated_hours": 0.0042}],
                    }
                )
                + "\n"
            )
            (eval_dir / "dense_eval_summary.json").write_text(
                json.dumps(
                    {
                        "sample_count": 1,
                        "split": "eval",
                        "checkpoint_summary": {"channel_mode": "rgb_aux"},
                        "aggregate": {
                            "precision@0.50_mean": 0.5,
                            "recall@0.50_mean": 0.25,
                            "fp@0.50_mean": 1.0,
                            "det_count_mean": 2.0,
                        },
                    }
                )
                + "\n"
            )

            rollup = build_training_rollup([export_dir, train_dir, eval_dir])
            self.assertEqual(rollup["run_count"], 3)
            self.assertEqual([run["kind"] for run in rollup["runs"]], ["export", "train_dense", "dense_eval"])
            self.assertEqual(rollup["runs"][1]["channel_mode"], "rgb_aux")
            self.assertEqual(rollup["runs"][1]["throughput_key"], "sample_epochs_per_second")
            self.assertAlmostEqual(rollup["runs"][2]["metrics"]["recall@0.50_mean"], 0.25)

            html_path = write_training_rollup(rollup, root / "rollup")
            html = html_path.read_text()
            self.assertIn("PerceptionISP RGB+Aux Training Rollup", html)
            self.assertIn("R@0.50", html)
            self.assertTrue((html_path.parent / "training_rollup_summary.json").exists())

    def test_rollup_cli_outputs_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "train"
            train_dir.mkdir()
            (train_dir / "train_smoke_summary.json").write_text(
                json.dumps(
                    {
                        "sample_count": 2,
                        "train_sample_count": 1,
                        "eval_sample_count": 1,
                        "epochs": 1,
                        "device": "cpu",
                        "elapsed_seconds": 0.5,
                        "sample_epochs_per_second": 2.0,
                    }
                )
                + "\n"
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = rollup_main([str(train_dir), "--output-dir", str(root / "rollup")])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["run_count"], 1)
            self.assertTrue((root / "rollup" / "training_rollup_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
