from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.casebook import build_casebook_from_path, write_casebook


class CasebookTest(unittest.TestCase):
    def test_build_casebook_selects_successes_and_counterexamples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = _write_report(root / "comparison")

            summary = build_casebook_from_path(
                report,
                baseline_input="human_rgb",
                target_input="perception_target",
                max_cases_per_category=2,
            )

            self.assertEqual(summary["sample_count"], 4)
            self.assertEqual(summary["aggregate"]["fp_delta_count"], -1)
            self.assertEqual(summary["categories"]["fp_reduction_success"]["case_count"], 1)
            self.assertEqual(summary["categories"]["recall_tradeoff"]["case_count"], 1)
            self.assertEqual(summary["categories"]["recall_loss_failure"]["case_count"], 1)
            self.assertEqual(summary["categories"]["fp_regression_failure"]["case_count"], 1)
            self.assertEqual(summary["status"], "pass")

            html_path = write_casebook(summary, root / "casebook")
            written = json.loads((html_path.parent / "casebook_summary.json").read_text())
            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Success/Failure Casebook", html_path.read_text())
            success_case = written["categories"]["fp_reduction_success"]["cases"][0]
            self.assertTrue(Path(success_case["visual_path"]).exists())


def _write_report(path: Path) -> Path:
    path.mkdir()
    image_path = path / "scene.png"
    image = np.full((64, 96, 3), 230, dtype=np.uint8)
    image[12:34, 12:34, :] = 80
    image[36:54, 56:84, :] = 120
    Image.fromarray(image).save(image_path)
    samples = [
        _sample("success", image_path, baseline=(_tp(), _fp()), target=(_tp(),)),
        _sample("tradeoff", image_path, baseline=(_tp(), _fp()), target=()),
        _sample("recall_loss", image_path, baseline=(_tp(),), target=()),
        _sample("fp_regression", image_path, baseline=(_tp(),), target=(_tp(), _fp())),
    ]
    payload = {
        "sample_count": len(samples),
        "run_config": {"label_agnostic": False},
        "samples": samples,
        "aggregate": {},
    }
    (path / "comparison_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _sample(sample_id: str, image_path: Path, *, baseline: tuple[dict, ...], target: tuple[dict, ...]) -> dict:
    gt = {"xyxy": [10, 10, 36, 36], "label": "person"}
    return {
        "sample_id": sample_id,
        "source": "unit",
        "ground_truth": [gt],
        "metadata": {
            "image_path": str(image_path),
            "width": 96,
            "height": 64,
            "original_width": 96,
            "original_height": 64,
            "cfa_pattern": "GRBG",
            "raw_provenance": {"true_sensor_cfa_mosaic": True, "pattern_remapped": False, "target_cfa_pattern": "GRBG"},
        },
        "detectors": [
            {"input_name": "human_rgb", "detections": list(baseline)},
            {"input_name": "perception_target", "detections": list(target)},
        ],
        "metrics": {
            "human_rgb": _metrics(baseline),
            "perception_target": _metrics(target),
        },
    }


def _metrics(detections: tuple[dict, ...]) -> dict:
    tp = sum(1 for row in detections if row["box"]["xyxy"][0] < 40)
    fp = len(detections) - tp
    fn = max(1 - tp, 0)
    return {
        "tp@0.50": tp,
        "fp@0.50": fp,
        "fn@0.50": fn,
        "precision@0.50": tp / max(tp + fp, 1),
        "recall@0.50": tp,
    }


def _tp() -> dict:
    return {
        "box": {"xyxy": [11, 11, 35, 35], "label": "person"},
        "score": 0.9,
        "metadata": {"fusion": {"edge_support": 0.8, "aux_support": 0.7}},
    }


def _fp() -> dict:
    return {
        "box": {"xyxy": [56, 36, 84, 54], "label": "person"},
        "score": 0.4,
        "metadata": {"fusion": {"edge_support": 0.2, "aux_support": 0.1}},
    }


if __name__ == "__main__":
    unittest.main()
