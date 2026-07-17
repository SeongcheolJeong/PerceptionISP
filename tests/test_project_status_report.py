from __future__ import annotations

import unittest

from perception_isp.cli import COMMANDS
from perception_isp.reporting.project_status_report import _render_html


class ProjectStatusReportTest(unittest.TestCase):
    def test_cli_exposes_status_report(self) -> None:
        self.assertEqual(
            COMMANDS[("report", "status")].module,
            "perception_isp.reporting.project_status_report",
        )

    def test_render_has_eight_accessible_tabs_and_algorithm_rationale(self) -> None:
        report = {
            "title": "fixture",
            "generated_at": "2026-07-17T00:00:00+00:00",
            "executive_conclusion": "fixture conclusion",
            "validation": {
                "pytest_passed": 1,
                "pytest_skipped": 0,
                "pytest_subtests_passed": 2,
                "camerae2e_python312_runtime_smoke": "pass",
            },
            "showcase": {
                "source_frames": 6,
                "pipeline_results": 11,
                "aux_maps_per_result": 33,
                "aux_map_artifacts": 363,
                "cache_hits": 0,
            },
            "software": {
                "core_python": "3.11+",
                "camerae2e_python": "3.12+",
                "aux_groups": {"hdr_exposure": 5},
                "perceptionisp_git": {"head": "abc", "dirty_count": 1, "path": "/repo"},
                "camerae2e_git": {"head": "def", "dirty_count": 1, "path": "/camera"},
            },
            "evidence": {
                "native_raw": {},
                "dense_aux_ablation": {},
                "scene_edge": {},
                "yolo_seed2": {},
                "hard_slice_seed3": {},
            },
            "milestones": [],
            "roadmap": [],
        }

        rendered = _render_html(report)

        self.assertEqual(rendered.count("role='tab'"), 8)
        self.assertEqual(rendered.count("role='tabpanel'"), 8)
        self.assertIn("1. 목적·판단기준", rendered)
        self.assertIn("2. 아키텍처", rendered)
        self.assertIn("3. 33 Aux 수식", rendered)
        self.assertIn("7. 다음 해야 할 일", rendered)
        self.assertIn("PerceptionISP end-to-end architecture", rendered)
        self.assertEqual(rendered.count("class='aux-card'"), 33)
        self.assertIn("Noise &amp; reliability · 3 maps", rendered)
        self.assertIn("E_conf=clip", rendered)
        self.assertIn("fitted probability가 아닙니다", rendered)
        self.assertIn("실제 streaming early-output path가 아닙니다", rendered)
        self.assertIn("nuScenes JPEG가 native RAW로 복원됐다", rendered)
        self.assertIn("16×H×W", rendered)
        self.assertIn("nuScenes Dataset Terms", rendered)


if __name__ == "__main__":
    unittest.main()
