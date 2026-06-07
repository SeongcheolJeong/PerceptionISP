from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.adverse_task_slice import (
    SUMMARY_FILENAME,
    build_adverse_task_slice,
    main as adverse_task_main,
    write_adverse_task_slice,
)


class AdverseTaskSliceTest(unittest.TestCase):
    def test_builds_condition_and_group_task_gate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adverse = _write_adverse_report(root / "adverse")

            summary = build_adverse_task_slice(
                json.loads((adverse / "adverse_native_slice_summary.json").read_text()),
                adverse_summary_path=adverse / "adverse_native_slice_summary.json",
                target_input="perception_target",
                baseline_input="human_rgb",
                profile="fp_reducer",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], "adverse_task_gate_partially_supported")
            self.assertEqual(summary["aggregate"]["adverse_passed_condition_count"], 2)
            self.assertEqual(summary["aggregate"]["adverse_condition_count"], 3)
            self.assertEqual(summary["conditions"][1]["condition"], "fog")
            self.assertTrue(summary["conditions"][1]["pass"])
            self.assertFalse(summary["conditions"][3]["pass"])
            groups = {row["group"]: row for row in summary["group_summary"]}
            self.assertEqual(groups["person"]["pass_condition_count"], 3)
            self.assertEqual(groups["small_all"]["fail_condition_count"], 1)

            html_path = write_adverse_task_slice(summary, root / "task")
            self.assertTrue((html_path.parent / SUMMARY_FILENAME).exists())
            html = html_path.read_text()
            self.assertIn("Adverse Task Slice", html)
            self.assertIn("adverse_task_gate_partially_supported", html)
            self.assertIn("small_all", html)

    def test_cli_outputs_compact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adverse = _write_adverse_report(root / "adverse")
            exit_code = adverse_task_main(
                [
                    str(adverse),
                    "--target-input",
                    "perception_target",
                    "--baseline-input",
                    "human_rgb",
                    "--output-dir",
                    str(root / "task"),
                ]
            )

            self.assertEqual(exit_code, 0)
            data = json.loads((root / "task" / SUMMARY_FILENAME).read_text())
            self.assertEqual(data["claim_status"], "adverse_task_gate_partially_supported")


def _write_adverse_report(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    runs = []
    for index, condition in enumerate(("nominal", "fog", "night", "hdr"), start=1):
        run_dir = path / f"{index:03d}_condition-{condition}"
        run_dir.mkdir()
        (run_dir / "index.html").write_text("<html></html>")
        _write_comparison(run_dir / "comparison_summary.json", condition=condition)
        runs.append(
            {
                "run_id": f"condition-{condition}",
                "condition": condition,
                "report": f"{index:03d}_condition-{condition}/index.html",
                "sample_count": 2,
                "raw_condition_summary": {
                    "sample_count": 2,
                    "true_sensor_cfa_mosaic_count": 2,
                    "pattern_remapped_count": 0,
                },
            }
        )
    payload = {
        "status": "pass",
        "claim_status": "adverse_fp_reducer_supported",
        "run_count": 4,
        "expected_run_count": 4,
        "cfa_pattern": "GRBG",
        "psf_sigma": 0.0,
        "runs": runs,
    }
    (path / "adverse_native_slice_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _write_comparison(path: Path, *, condition: str) -> None:
    target_detections = [_det("person", [0, 0, 20, 20])]
    if condition == "hdr":
        target_detections = []
    samples = [
        {
            "sample_id": f"{condition}_person",
            "ground_truth": [{"xyxy": [0, 0, 20, 20], "label": "person"}],
            "metadata": {"adverse_condition": condition},
            "detectors": [
                {"input_name": "human_rgb", "detections": [_det("person", [0, 0, 20, 20]), _det("person", [40, 40, 50, 50])]},
                {"input_name": "perception_target", "detections": target_detections},
            ],
        },
        {
            "sample_id": f"{condition}_vehicle",
            "ground_truth": [{"xyxy": [0, 0, 80, 80], "label": "car"}],
            "metadata": {"adverse_condition": condition},
            "detectors": [
                {"input_name": "human_rgb", "detections": [_det("car", [0, 0, 80, 80]), _det("car", [90, 90, 110, 110])]},
                {"input_name": "perception_target", "detections": [_det("car", [0, 0, 80, 80])]},
            ],
        },
    ]
    payload = {
        "sample_count": 2,
        "run_config": {"label_agnostic": False},
        "aggregate": {"human_rgb": {}, "perception_target": {}},
        "samples": samples,
    }
    path.write_text(json.dumps(payload) + "\n")


def _det(label: str, xyxy: list[float]) -> dict:
    return {"box": {"xyxy": xyxy, "label": label}, "score": 0.9}


if __name__ == "__main__":
    unittest.main()
