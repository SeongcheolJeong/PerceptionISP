from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.camerae2e_bridge import raw_from_rgb_direct
from perception_isp.eval_types import EvaluationSample
from perception_isp.scene_edge_aux_sweep import (
    SUMMARY_FILENAME,
    build_scene_edge_aux_sweep,
    main as aux_sweep_main,
    write_scene_edge_aux_sweep,
)
from perception_isp.synthetic import make_synthetic_scene_rgb


class SceneEdgeAuxSweepTest(unittest.TestCase):
    def test_build_scene_edge_aux_sweep_compares_candidate_maps(self) -> None:
        scene_rgb = make_synthetic_scene_rgb(width=96, height=64, frame_counter=0, seed=707)
        sample = EvaluationSample(
            sample_id="aux_sweep",
            raw=raw_from_rgb_direct(scene_rgb, width=48, height=32, cfa_pattern="RGGB"),
            ground_truth=(),
            source="unit",
            reference_rgb=scene_rgb,
        )

        summary = build_scene_edge_aux_sweep((sample,))

        self.assertIn(summary["status"], {"pass", "warning"})
        self.assertEqual(summary["case_count"], 1)
        self.assertIn("edge_confidence", summary["aggregate"])
        self.assertIn("sqrt_norm_conf_strength", summary["aggregate"])
        self.assertIn(summary["best_candidate"]["name"], summary["aggregate"])
        self.assertTrue(all(row["status"] in {"pass", "fail"} for row in summary["checks"]))

    def test_write_scene_edge_aux_sweep_outputs_report_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_rgb = make_synthetic_scene_rgb(width=96, height=64, frame_counter=1, seed=808)
            image_path = root / "scene.png"
            from PIL import Image

            Image.fromarray((scene_rgb * 255.0).round().astype("uint8")).save(image_path)
            sample = EvaluationSample(
                sample_id="write_aux_sweep",
                raw=raw_from_rgb_direct(scene_rgb, width=48, height=32, cfa_pattern="RGGB"),
                ground_truth=(),
                source="unit",
                reference_rgb=scene_rgb,
            )
            html_path = write_scene_edge_aux_sweep(build_scene_edge_aux_sweep((sample,)), root / "report")

            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Scene-Edge Aux Sweep", html_path.read_text())
            persisted = json.loads((html_path.parent / SUMMARY_FILENAME).read_text())
            self.assertIn(persisted["status"], {"pass", "warning"})

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = aux_sweep_main(
                    [
                        "--source",
                        "sample-image",
                        "--image-path",
                        str(image_path),
                        "--width",
                        "48",
                        "--height",
                        "32",
                        "--scene-scale",
                        "2",
                        "--no-camerae2e",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertIn(printed["status"], {"pass", "warning"})
            self.assertTrue((root / "cli" / SUMMARY_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
