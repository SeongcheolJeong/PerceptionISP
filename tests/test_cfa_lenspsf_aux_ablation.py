from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.cfa_lenspsf_aux_ablation import build_aux_ablation, main as aux_ablation_main, write_aux_ablation


class CfaLensPsfAuxAblationTest(unittest.TestCase):
    def test_build_aux_ablation_records_recall_fp_tradeoff(self) -> None:
        no_aux = _sweep("perception_calibrated_score_label_fusion_rgb_aux", fp_values=(1.0, 1.2))
        aux = _sweep("perception_calibrated_score_label_aux_fusion_rgb_aux_t001", fp_values=(1.1, 1.3), recall_values=(0.42, 0.43))

        summary = build_aux_ablation(no_aux, aux)

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["claim_status"], "aux_recall_fp_tradeoff")
        self.assertEqual(summary["condition_count"], 2)
        self.assertEqual(summary["aggregate"]["aux_recall_win_count"], 2)
        self.assertEqual(summary["aggregate"]["aux_fp_win_count"], 0)
        self.assertAlmostEqual(summary["aggregate"]["mean_aux_minus_no_aux_fp@0.50"], 0.1)
        self.assertEqual(summary["checks"][2]["status"], "warning")

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no_aux_dir = _write_sweep(root / "no_aux", _sweep("perception_calibrated_score_label_fusion_rgb_aux"))
            aux_dir = _write_sweep(root / "aux", _sweep("perception_calibrated_score_label_aux_fusion_rgb_aux_t001", fp_values=(0.9, 1.0)))

            summary = build_aux_ablation(json.loads((no_aux_dir / "cfa_lenspsf_detector_sweep_summary.json").read_text()), json.loads((aux_dir / "cfa_lenspsf_detector_sweep_summary.json").read_text()))
            html_path = write_aux_ablation(summary, root / "report")

            self.assertTrue(html_path.exists())
            self.assertTrue((html_path.parent / "cfa_lenspsf_aux_ablation_summary.json").exists())
            self.assertIn("CFA/LensPSF Score-Label Aux Ablation", html_path.read_text())

            rc = aux_ablation_main([str(no_aux_dir), str(aux_dir), "--output-dir", str(root / "cli_report")])
            self.assertEqual(rc, 0)
            self.assertTrue((root / "cli_report" / "cfa_lenspsf_aux_ablation_summary.json").exists())


def _sweep(input_name: str, *, fp_values: tuple[float, float] = (1.0, 1.2), recall_values: tuple[float, float] = (0.40, 0.41)) -> dict:
    return {
        "status": "pass",
        "run_count": 2,
        "expected_run_count": 2,
        "count": 8,
        "runs": [
            _run("cfa-grbg_psf-0p00", "GRBG", 0.0, input_name, fp_values[0], recall_values[0]),
            _run("cfa-rggb_psf-1p60", "RGGB", 1.6, input_name, fp_values[1], recall_values[1]),
        ],
    }


def _run(run_id: str, cfa: str, psf: float, input_name: str, fp: float, recall: float) -> dict:
    metrics = {
        "precision@0.50_mean": 0.6 - fp * 0.01,
        "recall@0.50_mean": recall,
        "recall@0.75_mean": recall - 0.1,
        "small_recall@0.50_mean": recall - 0.2,
        "fp@0.50_mean": fp,
        "det_count_mean": fp + 2.0,
    }
    return {
        "run_id": run_id,
        "cfa_pattern": cfa,
        "psf_sigma": psf,
        "sample_count": 8,
        "metrics": {
            "human_rgb": metrics,
            input_name: metrics,
        },
        "delta_vs_human": {
            input_name: {
                "precision@0.50_mean": metrics["precision@0.50_mean"] - 0.5,
                "recall@0.50_mean": metrics["recall@0.50_mean"] - 0.4,
                "recall@0.75_mean": metrics["recall@0.75_mean"] - 0.3,
                "small_recall@0.50_mean": metrics["small_recall@0.50_mean"] - 0.2,
                "fp@0.50_mean": metrics["fp@0.50_mean"] - 1.2,
                "det_count_mean": metrics["det_count_mean"] - 3.2,
            }
        },
    }


def _write_sweep(path: Path, payload: dict) -> Path:
    path.mkdir()
    (path / "cfa_lenspsf_detector_sweep_summary.json").write_text(json.dumps(payload) + "\n")
    return path


if __name__ == "__main__":
    unittest.main()
