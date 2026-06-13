from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.dense_select_test_gate import (
    parse_int_list,
    select_candidate,
    split_indices_by_hash,
    write_dense_select_test_gate,
)


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
