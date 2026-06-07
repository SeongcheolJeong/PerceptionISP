from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.cfa_lenspsf_native_audit import build_native_audit_from_path, write_native_audit


class CfaLensPsfNativeAuditTest(unittest.TestCase):
    def test_native_audit_separates_native_and_remapped_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = _write_sweep(root / "sweep")

            summary = build_native_audit_from_path(sweep)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["groups"]["native"]["run_count"], 1)
            self.assertEqual(summary["groups"]["remapped"]["run_count"], 1)
            self.assertEqual(summary["groups"]["native"]["cfa_patterns"], ["GRBG"])
            self.assertEqual(summary["groups"]["remapped"]["cfa_patterns"], ["RGGB"])
            self.assertAlmostEqual(summary["groups"]["native"]["mean_delta_fp@0.50"], -0.4)
            self.assertAlmostEqual(summary["groups"]["remapped"]["mean_delta_fp@0.50"], -0.2)
            self.assertEqual(summary["runs"][0]["native_status"], "native")
            self.assertEqual(summary["runs"][1]["native_status"], "remapped")

            html_path = write_native_audit(summary, root / "audit")
            self.assertTrue(html_path.exists())
            self.assertTrue((html_path.parent / "cfa_lenspsf_native_audit_summary.json").exists())
            self.assertIn("Native-CFA Audit", html_path.read_text())

    def test_all_native_sweep_passes_without_remapped_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = _write_sweep(
                root / "sweep",
                runs=[
                    _run("cfa-grbg_psf-0p00", "GRBG", 0, 0.0, -0.4),
                    _run("cfa-rggb_psf-0p00", "RGGB", 0, 0.0, -0.2, source_cfa="RGGB"),
                ],
            )

            summary = build_native_audit_from_path(sweep)

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["groups"]["native"]["run_count"], 2)
            self.assertEqual(summary["groups"]["remapped"]["run_count"], 0)
            self.assertEqual(summary["checks"][2]["status"], "pass")


def _write_sweep(path: Path, *, runs: list[dict] | None = None) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html></html>")
    payload_runs = runs or [
        _run("cfa-grbg_psf-0p00", "GRBG", 0, 0.0, -0.4),
        _run("cfa-rggb_psf-0p00", "RGGB", 4, 1.0, -0.2),
    ]
    payload = {
        "run_count": len(payload_runs),
        "cfa_patterns": sorted({str(row["cfa_pattern"]) for row in payload_runs}),
        "psf_sigmas": [0.0],
        "runs": payload_runs,
    }
    (path / "cfa_lenspsf_detector_sweep_summary.json").write_text(json.dumps(payload) + "\n")
    return path


def _run(run_id: str, cfa: str, remapped: int, remapped_fraction: float, fp_delta: float, *, source_cfa: str = "GRBG") -> dict:
    return {
        "run_id": run_id,
        "report": f"{run_id}/index.html",
        "cfa_pattern": cfa,
        "psf_sigma": 0.0,
        "sample_count": 4,
        "raw_condition_summary": {
            "sample_count": 4,
            "source_cfa_patterns": {source_cfa: 4},
            "target_cfa_patterns": {cfa: 4},
            "pattern_remapped_count": remapped,
            "pattern_remapped_fraction": remapped_fraction,
        },
        "metrics": {
            "human_rgb": {"fp@0.50_mean": 1.0},
            "perception_calibrated_score_label_aux_fusion_rgb_aux_t001": {
                "precision@0.50_mean": 0.6,
                "recall@0.50_mean": 0.5,
                "small_recall@0.50_mean": 0.2,
                "fp@0.50_mean": 1.0 + fp_delta,
            },
        },
        "delta_vs_human": {
            "perception_calibrated_score_label_aux_fusion_rgb_aux_t001": {
                "precision@0.50_mean": 0.1,
                "recall@0.50_mean": 0.0,
                "small_recall@0.50_mean": 0.0,
                "fp@0.50_mean": fp_delta,
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
