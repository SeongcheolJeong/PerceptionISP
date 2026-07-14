from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from perception_isp.evaluation import dense_select_test_gate as gate
from perception_isp.evaluation.dense_select_test_gate import (
    build_dense_select_test_gate,
    parse_int_list,
    parse_optional_float_list,
    parse_optional_int_list,
    select_candidate,
    split_indices_by_hash,
    write_dense_select_test_gate,
)
from perception_isp.core.task_types import BoundingBox, Detection


class DenseSelectTestGateTest(unittest.TestCase):
    def test_split_indices_by_hash_is_deterministic_and_disjoint(self) -> None:
        selection, test = split_indices_by_hash(range(10), selection_fraction=0.5, split_seed="unit")
        selection_again, test_again = split_indices_by_hash(range(10), selection_fraction=0.5, split_seed="unit")
        self.assertEqual(selection, selection_again)
        self.assertEqual(test, test_again)
        self.assertEqual(len(selection), 5)
        self.assertEqual(len(test), 5)
        self.assertEqual(set(selection) & set(test), set())
        self.assertEqual(set(selection) | set(test), set(range(10)))

    def test_select_candidate_prefers_valid_recall_gain(self) -> None:
        selected, status = select_candidate(
            [
                {"epoch": 0, "confidence": 0.1, "metrics": {"fp": 10.0}, "deltas": {"precision": 0.04, "recall": 0.12, "fp": 0.2}},
                {"epoch": 1, "confidence": 0.2, "metrics": {"fp": 8.0}, "deltas": {"precision": 0.01, "recall": 0.08, "fp": -1.0}},
                {"epoch": 2, "confidence": 0.2, "metrics": {"fp": 9.0}, "deltas": {"precision": 0.02, "recall": 0.10, "fp": 0.0}},
            ]
        )
        self.assertEqual(status, "pass")
        self.assertEqual(selected["epoch"], 2)

    def test_parse_int_list_accepts_ranges(self) -> None:
        self.assertEqual(parse_int_list("0-2,4,2"), (0, 1, 2, 4))
        self.assertEqual(parse_int_list("3-1"), (3, 2, 1))
        self.assertEqual(parse_optional_float_list("none,0.4"), (None, 0.4))
        self.assertEqual(parse_optional_int_list("none,10"), (None, 10))

    def test_cached_metric_sweep_applies_threshold_and_topk(self) -> None:
        cache = {
            "samples": [
                {
                    "ground_truth": (BoundingBox((0.0, 0.0, 10.0, 10.0), label="car"),),
                    "detections": (
                        Detection(BoundingBox((0.0, 0.0, 10.0, 10.0), label="car"), score=0.9),
                        Detection(BoundingBox((20.0, 20.0, 30.0, 30.0), label="car"), score=0.8),
                    ),
                },
                {
                    "ground_truth": (BoundingBox((0.0, 0.0, 10.0, 10.0), label="car"),),
                    "detections": (
                        Detection(BoundingBox((20.0, 20.0, 30.0, 30.0), label="car"), score=0.7),
                    ),
                },
            ]
        }

        metrics = gate._eval_metrics_from_cache(cache, confidence=0.65, nms_iou=None, max_detections=1)

        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["fp"], 0.5)
        self.assertAlmostEqual(metrics["det_count"], 1.0)

    def test_build_gate_can_tune_rgb_baseline_threshold(self) -> None:
        def fake_eval(**kwargs):
            checkpoint = str(kwargs["checkpoint_path"])
            confidence = float(kwargs["confidence"])
            is_aux = "aux" in checkpoint
            if is_aux:
                recall = 0.80 if confidence >= 0.2 else 0.40
                precision = 0.25 if confidence >= 0.2 else 0.10
                fp = 9.0 if confidence >= 0.2 else 4.0
                if confidence < 0.2:
                    recall = 0.65
                    precision = 0.22
            else:
                recall = 0.60 if confidence >= 0.2 else 0.50
                precision = 0.20 if confidence >= 0.2 else 0.10
                fp = 5.0 if confidence >= 0.2 else 10.0
            return {
                "aggregate": {
                    "precision@0.50_mean": precision,
                    "recall@0.50_mean": recall,
                    "fp@0.50_mean": fp,
                    "det_count_mean": fp + 1.0,
                    "small_recall@0.50_mean": 0.0,
                }
            }

        with mock.patch.object(gate, "evaluate_dense_manifest", side_effect=fake_eval):
            summary = build_dense_select_test_gate(
                manifest_path="manifest.jsonl",
                source_eval_indices=(0, 1, 2, 3),
                seeds=(7,),
                rgb_only_template="rgb_{seed}.pt",
                aux_checkpoint_template="aux_{seed}_{epoch:03d}.pt",
                epochs=(0,),
                thresholds=(0.0, 0.2),
                rgb_thresholds=(0.0, 0.2),
                nms_ious=(None, 0.4),
                max_detections=(None, 10),
                tune_rgb_baseline=True,
            )
        row = summary["rows"][0]
        self.assertTrue(row["tune_rgb_baseline"])
        self.assertEqual(row["selected_rgb_confidence"], 0.2)
        self.assertEqual(row["selected_confidence"], 0.2)
        self.assertIn(row["selected_nms_iou"], (None, 0.4))
        self.assertIn(row["selected_max_detections"], (None, 10))
        self.assertEqual(row["selection_fp_budget"], 10.0)
        self.assertAlmostEqual(row["test_deltas"]["recall"], 0.20)

        with mock.patch.object(gate, "evaluate_dense_manifest", side_effect=fake_eval):
            strict = build_dense_select_test_gate(
                manifest_path="manifest.jsonl",
                source_eval_indices=(0, 1, 2, 3),
                seeds=(7,),
                rgb_only_template="rgb_{seed}.pt",
                aux_checkpoint_template="aux_{seed}_{epoch:03d}.pt",
                epochs=(0,),
                thresholds=(0.0, 0.2),
                rgb_thresholds=(0.0, 0.2),
                tune_rgb_baseline=True,
                aux_fp_budget_source="selected_rgb",
            )
        strict_row = strict["rows"][0]
        self.assertEqual(strict["claim_status"], "strict_fair_tuned_rgb_baseline_heldout_pass")
        self.assertEqual(strict_row["selection_fp_budget"], 5.0)
        self.assertTrue(strict_row["selection_aux_within_fp_budget"])
        self.assertEqual(strict_row["aux_fp_budget_source"], "selected_rgb")
        self.assertEqual(strict_row["selected_confidence"], 0.0)
        self.assertAlmostEqual(strict_row["test_deltas"]["fp"], -1.0)

    def test_build_gate_thread_backend_reports_sorted_rows(self) -> None:
        def fake_eval(**kwargs):
            checkpoint = str(kwargs["checkpoint_path"])
            is_aux = "aux" in checkpoint
            return {
                "aggregate": {
                    "precision@0.50_mean": 0.30 if is_aux else 0.20,
                    "recall@0.50_mean": 0.70 if is_aux else 0.60,
                    "fp@0.50_mean": 4.0 if is_aux else 5.0,
                    "det_count_mean": 5.0 if is_aux else 6.0,
                    "small_recall@0.50_mean": 0.0,
                }
            }

        with mock.patch.object(gate, "evaluate_dense_manifest", side_effect=fake_eval):
            summary = build_dense_select_test_gate(
                manifest_path="manifest.jsonl",
                source_eval_indices=(0, 1, 2, 3),
                seeds=(8, 7),
                rgb_only_template="rgb_{seed}.pt",
                aux_checkpoint_template="aux_{seed}_{epoch:03d}.pt",
                epochs=(0,),
                thresholds=(0.0,),
                rgb_thresholds=(0.0,),
                nms_ious=(None,),
                max_detections=(None,),
                jobs=2,
                job_backend="thread",
                progress=True,
            )

        self.assertEqual(summary["job_backend"], "thread")
        self.assertEqual(summary["jobs"], 2)
        self.assertEqual([row["seed"] for row in summary["rows"]], [7, 8])
        self.assertEqual(summary["pass_test_seed_count"], 2)

    def test_write_dense_select_test_gate_outputs_report_files(self) -> None:
        summary = {
            "claim_status": "heldout_test_equal_fp_3seed_improvement",
            "pass_test_seed_count": 1,
            "seed_count": 1,
            "source_eval_count": 4,
            "selection_sample_count": 2,
            "test_sample_count": 2,
            "selection_indices": [0, 1],
            "test_indices": [2, 3],
            "candidate_epochs": [0],
            "candidate_thresholds": [0.1],
            "mean_test_deltas": {"precision": 0.1, "recall": 0.2, "fp": -1.0},
            "rows": [
                {
                    "seed": 7,
                    "selected_epoch": 0,
                    "selected_confidence": 0.1,
                    "selected_rgb_confidence": 0.0,
                    "test_status": "pass",
                    "test_rgb_only": {"precision": 0.1, "recall": 0.2, "fp": 5.0},
                    "test_rgb_aux": {"precision": 0.2, "recall": 0.4, "fp": 4.0},
                    "test_deltas": {"precision": 0.1, "recall": 0.2, "fp": -1.0},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            html_path = write_dense_select_test_gate(summary, tmp)
            self.assertTrue(html_path.exists())
            self.assertTrue((Path(tmp) / "summary.json").exists())
            split = json.loads((Path(tmp) / "split_indices.json").read_text())
            self.assertEqual(split["selection_indices"], [0, 1])
            self.assertIn("PerceptionISP Selection/Test RGB+Aux Gate", html_path.read_text())


if __name__ == "__main__":
    unittest.main()
