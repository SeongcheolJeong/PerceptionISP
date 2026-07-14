from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from perception_isp.datasets.raw_dataset_readiness import (
    build_raw_dataset_readiness,
    main as raw_dataset_readiness_main,
    write_raw_dataset_readiness,
)


class RawDatasetReadinessTest(unittest.TestCase):
    def test_build_report_prioritizes_real_adverse_raw_datasets(self) -> None:
        summary = build_raw_dataset_readiness()

        self.assertEqual(summary["status"], "pass")
        datasets = {row["name"]: row for row in summary["datasets"]}
        self.assertEqual(datasets["AODRaw"]["priority"], "P0")
        self.assertEqual(datasets["ROD / RAOD"]["priority"], "P0")
        self.assertIn("adverse_native_raw_slice", datasets["AODRaw"]["strong_for"])
        self.assertIn("automotive_hdr_claim", datasets["ROD / RAOD"]["strong_for"])
        self.assertIn("Synthetic-only evidence", datasets["ADE20K-RAW"]["risk"])
        adverse = next(row for row in summary["claim_priorities"] if row["claim"] == "Adverse-condition/native RAW slice")
        self.assertEqual(adverse["best_datasets"][:3], ["AODRaw", "ROD / RAOD", "LOD"])
        self.assertIn("Do not bulk-download", summary["recommended_sequence"][0])

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = build_raw_dataset_readiness()
            html_path = write_raw_dataset_readiness(summary, root / "report")
            self.assertTrue((html_path.parent / "raw_dataset_readiness_summary.json").exists())
            html = html_path.read_text()
            self.assertIn("PerceptionISP RAW Dataset Readiness", html)
            self.assertIn("AODRaw", html)
            self.assertIn("ROD / RAOD", html)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = raw_dataset_readiness_main(["--output-dir", str(root / "cli")])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertIn("AODRaw", printed["top_priority"])
            self.assertTrue((root / "cli" / "raw_dataset_readiness_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
