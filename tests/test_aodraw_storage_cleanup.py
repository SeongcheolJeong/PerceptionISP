from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from perception_isp.datasets.aodraw_storage_cleanup import (
    CONFIRM_TOKEN,
    build_aodraw_storage_cleanup,
    main as aodraw_storage_cleanup_main,
    write_aodraw_storage_cleanup,
)


class AODRawStorageCleanupTest(unittest.TestCase):
    def test_dry_run_selects_first_sufficient_candidate_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "exports" / "large"
            _write(candidate / "blob.bin", b"x" * 1024)

            with _patch_cleanup(candidate):
                summary = build_aodraw_storage_cleanup(
                    project_root=root,
                    dataset_root=root,
                    execute=False,
                )

            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["selected_candidate_count"], 1)
            self.assertEqual(summary["actions"][0]["status"], "would_delete")
            self.assertTrue(candidate.exists())
            self.assertIn("--confirm-token", summary["execute_command"])

    def test_execute_requires_confirmation_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "exports" / "large"
            _write(candidate / "blob.bin", b"x" * 1024)

            with _patch_cleanup(candidate):
                summary = build_aodraw_storage_cleanup(
                    project_root=root,
                    dataset_root=root,
                    execute=True,
                    confirm_token="wrong",
                )

            self.assertEqual(summary["status"], "blocked_confirmation_required")
            self.assertEqual(summary["actions"][0]["status"], "blocked_confirmation_required")
            self.assertTrue(candidate.exists())

    def test_execute_deletes_confirmed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "exports" / "large"
            _write(candidate / "blob.bin", b"x" * 1024)

            with _patch_cleanup(candidate):
                summary = build_aodraw_storage_cleanup(
                    project_root=root,
                    dataset_root=root,
                    execute=True,
                    confirm_token=CONFIRM_TOKEN,
                )

            self.assertEqual(summary["status"], "executed")
            self.assertEqual(summary["actions"][0]["status"], "deleted")
            self.assertFalse(candidate.exists())

    def test_write_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "exports" / "large"
            _write(candidate / "blob.bin", b"x" * 1024)

            with _patch_cleanup(candidate):
                summary = build_aodraw_storage_cleanup(project_root=root, dataset_root=root)
            html_path = write_aodraw_storage_cleanup(summary, root / "report")
            self.assertTrue((html_path.parent / "aodraw_storage_cleanup_summary.json").exists())
            self.assertIn("AODRaw Storage Cleanup", html_path.read_text())

            stdout = io.StringIO()
            with _patch_cleanup(candidate):
                with contextlib.redirect_stdout(stdout):
                    exit_code = aodraw_storage_cleanup_main(
                        [
                            "--project-root",
                            str(root),
                            "--dataset-root",
                            str(root),
                            "--output-dir",
                            str(root / "cli"),
                        ]
                    )
            printed = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(printed["status"], "dry_run")
            self.assertTrue((root / "cli" / "aodraw_storage_cleanup_summary.json").exists())


def _patch_cleanup(candidate: Path):
    row = {
        "path": str(candidate.relative_to(candidate.parents[1])),
        "absolute_path": str(candidate),
        "label": "unit large export",
        "risk": "low",
        "why": "unit cleanup candidate",
        "size_gib": 12.0,
        "size_bytes": int(12 * 1024**3),
        "manual_action": "unit",
    }
    return mock.patch.multiple(
        "perception_isp.datasets.aodraw_storage_cleanup",
        _cleanup_candidates=mock.Mock(return_value=[row]),
        _cleanup_summary=mock.Mock(
            return_value={
                "raw_test_required_with_headroom_gib": 64.89,
                "additional_free_gib_needed": 10.0,
                "candidate_count": 1,
                "candidate_total_gib": 12.0,
                "cleanup_can_cover_gap": True,
                "first_sufficient_candidate": row["path"],
                "note": "unit",
            }
        ),
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


if __name__ == "__main__":
    unittest.main()
