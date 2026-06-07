from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.camerae2e_bridge import raw_from_rgb_direct
from perception_isp.eval_types import EvaluationSample
from perception_isp.scene_edge_confidence_suite import (
    SCENE_EDGE_CONFIDENCE_SUMMARY,
    build_scene_edge_confidence_suite,
    main as scene_edge_main,
    write_scene_edge_confidence_suite,
)
from perception_isp.synthetic import make_synthetic_scene_rgb


class SceneEdgeConfidenceSuiteTest(unittest.TestCase):
    def test_build_scene_edge_confidence_suite_compares_high_info_scene_edges(self) -> None:
        scene_rgb = make_synthetic_scene_rgb(width=96, height=64, frame_counter=0, seed=101)
        raw = raw_from_rgb_direct(scene_rgb, width=48, height=32, cfa_pattern="RGGB")
        sample = EvaluationSample(
            sample_id="high_info_scene",
            raw=raw,
            ground_truth=(),
            source="unit_high_info_scene",
            metadata={"scene_width": 96, "scene_height": 64, "width": 48, "height": 32},
            reference_rgb=scene_rgb,
        )

        summary = build_scene_edge_confidence_suite((sample,))

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["sample_count"], 1)
        checks = {row["id"]: row for row in summary["checks"]}
        self.assertEqual(checks["finite_scene_edge_outputs"]["status"], "pass")
        self.assertEqual(checks["reference_scene_edges_present"]["status"], "pass")
        self.assertEqual(checks["human_and_perception_edges_track_scene_edges"]["status"], "pass")
        case = summary["cases"][0]
        metrics = case["metrics"]
        self.assertGreater(metrics["source_edge_fraction"], 0.005)
        self.assertGreater(metrics["human_rgb_proxy_scene_edge_separation"], 0.0)
        self.assertGreater(metrics["perception_aux_confidence_scene_edge_separation"], 0.0)

    def test_write_scene_edge_confidence_suite_outputs_report_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_rgb = make_synthetic_scene_rgb(width=80, height=48, frame_counter=0, seed=202)
            raw = raw_from_rgb_direct(scene_rgb, width=40, height=24, cfa_pattern="RGGB")
            sample = EvaluationSample(
                sample_id="write_scene",
                raw=raw,
                ground_truth=(),
                source="unit_scene",
                reference_rgb=scene_rgb,
            )
            summary = build_scene_edge_confidence_suite((sample,))
            html_path = write_scene_edge_confidence_suite(summary, root / "report")
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Scene Edge Confidence", html_path.read_text())
            persisted = json.loads((html_path.parent / SCENE_EDGE_CONFIDENCE_SUMMARY).read_text())
            self.assertEqual(persisted["status"], "pass")
            self.assertNotIn("_assets_source", persisted["cases"][0])
            self.assertGreater(len(list((html_path.parent / "assets").glob("*.png"))), 0)

            image_path = root / "scene.png"
            from PIL import Image

            Image.fromarray((scene_rgb * 255.0).round().astype("uint8")).save(image_path)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = scene_edge_main(
                    [
                        "--source",
                        "sample-image",
                        "--image-path",
                        str(image_path),
                        "--width",
                        "40",
                        "--height",
                        "24",
                        "--scene-scale",
                        "2",
                        "--no-camerae2e",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["failed_checks"], [])
            self.assertTrue((root / "cli" / SCENE_EDGE_CONFIDENCE_SUMMARY).exists())


if __name__ == "__main__":
    unittest.main()
