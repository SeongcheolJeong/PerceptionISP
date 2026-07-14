from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from perception_isp.cli import COMMANDS, main


class UnifiedCliTest(unittest.TestCase):
    def test_top_level_help_lists_workflow_groups(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--help"]), 0)
        rendered = stdout.getvalue()
        for group in ("isp:", "data:", "train:", "evaluate:", "report:"):
            self.assertIn(group, rendered)

    def test_dispatch_forwards_remaining_arguments(self) -> None:
        entrypoint = mock.Mock(return_value=0)
        module = mock.Mock(main=entrypoint)
        with mock.patch("perception_isp.cli.importlib.import_module", return_value=module) as importer:
            self.assertEqual(main(["isp", "run", "--width", "64"]), 0)
        importer.assert_called_once_with(COMMANDS[("isp", "run")].module)
        entrypoint.assert_called_once_with(["--width", "64"])

    def test_unknown_command_returns_usage_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["train", "missing"]), 2)
        self.assertIn("Unknown command", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
