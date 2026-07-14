from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception_isp.reporting.scene_edge_casebook import (
    SUMMARY_FILENAME,
    build_scene_edge_casebook_from_path,
    main as casebook_main,
    write_scene_edge_casebook,
)


class SceneEdgeCasebookTest(unittest.TestCase):
    def test_build_scene_edge_casebook_selects_successes_and_counterexamples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_scene_edge_summary(root / "scene_edge")

            summary = build_scene_edge_casebook_from_path(source, max_cases_per_category=4)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["source_case_count"], 3)
            self.assertGreater(summary["selected_case_count"], 0)
            categories = summary["categories"]
            self.assertGreater(categories["rgb_edge_improvement"]["selected_case_count"], 0)
            self.assertGreater(categories["aux_confidence_success"]["selected_case_count"], 0)
            self.assertGreater(categories["aux_confidence_counterexample"]["selected_case_count"], 0)
            self.assertGreater(categories["rgb_edge_regression"]["selected_case_count"], 0)

    def test_write_scene_edge_casebook_outputs_report_assets_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_scene_edge_summary(root / "scene_edge")
            summary = build_scene_edge_casebook_from_path(source)
            html_path = write_scene_edge_casebook(summary, root / "casebook")

            self.assertTrue(html_path.exists())
            self.assertIn("PerceptionISP Scene-Edge Casebook", html_path.read_text())
            persisted = json.loads((html_path.parent / SUMMARY_FILENAME).read_text())
            self.assertEqual(persisted["status"], "pass")
            self.assertGreater(len(list((html_path.parent / "assets").glob("*.png"))), 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = casebook_main(
                    [
                        str(source),
                        "--max-cases-per-category",
                        "2",
                        "--output-dir",
                        str(root / "cli"),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "pass")
            self.assertTrue((root / "cli" / SUMMARY_FILENAME).exists())


def _write_scene_edge_summary(path: Path) -> Path:
    path.mkdir()
    assets = path / "assets"
    assets.mkdir()
    asset_names = {}
    for name in ("reference_rgb", "source_edge", "human_rgb", "perception_rgb", "aux_edge_confidence", "aux_edge_strength"):
        filename = f"{name}.png"
        _write_asset(assets / filename)
        asset_names[name] = f"assets/{filename}"
    payload = {
        "status": "fail",
        "checks": [{"id": "human_and_perception_edges_track_scene_edges", "status": "fail"}],
        "cases": [
            _case("success", rgb_delta=0.04, aux_delta=0.25, aux_sep=0.05, assets=asset_names),
            _case("counterexample", rgb_delta=0.03, aux_delta=-0.30, aux_sep=-0.01, assets=asset_names),
            _case("rgb_regression", rgb_delta=-0.02, aux_delta=-0.10, aux_sep=0.02, assets=asset_names),
        ],
    }
    (path / "index.html").write_text("<html></html>")
    summary = path / "scene_edge_confidence_summary.json"
    summary.write_text(json.dumps(payload) + "\n")
    return summary


def _case(case_id: str, *, rgb_delta: float, aux_delta: float, aux_sep: float, assets: dict[str, str]) -> dict:
    human = 0.50
    return {
        "id": case_id,
        "source": "unit",
        "cfa_pattern": "RGGB",
        "psf_sigma": 0.0,
        "assets": dict(assets),
        "metrics": {
            "source_edge_fraction": 0.10,
            "human_rgb_proxy_source_edge_f1": human,
            "perception_rgb_proxy_source_edge_f1": human + rgb_delta,
            "perception_rgb_minus_human_source_edge_f1": rgb_delta,
            "perception_aux_confidence_source_edge_f1": human + aux_delta,
            "perception_aux_confidence_minus_human_source_edge_f1": aux_delta,
            "perception_aux_confidence_scene_edge_separation": aux_sep,
            "perception_aux_strength_source_edge_f1": 0.25,
            "perception_aux_strength_minus_human_source_edge_f1": -0.25,
        },
    }


def _write_asset(path: Path) -> None:
    arr = np.full((12, 16, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    unittest.main()
