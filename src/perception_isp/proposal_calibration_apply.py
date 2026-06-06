"""Apply a saved proposal calibration model to a comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .comparison import write_comparison_report
from .proposal_calibration import apply_proposal_calibration_to_report
from .types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Apply proposal calibration model artifact to a saved comparison report.")
    parser.add_argument("report", help="Report directory or comparison_summary.json")
    parser.add_argument("--model", required=True, help="proposal_calibration_model.json")
    parser.add_argument("--split", default="all", choices=["all", "train", "eval"], help="Apply to all samples or artifact train/eval indices.")
    parser.add_argument("--output-input", default=None, help="Override output input name.")
    parser.add_argument("--output-dir", default="reports/perception_proposal_calibration_applied")
    args = parser.parse_args(argv)

    report_path = _summary_path(args.report)
    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"proposal calibration model not found: {model_path}")
    report = json.loads(report_path.read_text())
    artifact = json.loads(model_path.read_text())
    indices = _indices_for_split(artifact, str(args.split))
    result = apply_proposal_calibration_to_report(
        report,
        artifact,
        output_input_name=args.output_input,
        indices=indices,
    )
    result["run_config"]["proposal_calibration"]["model_path"] = str(model_path)
    result["run_config"]["proposal_calibration"]["split"] = str(args.split)
    html_path = write_comparison_report(result, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / "comparison_summary.json"),
                    "aggregate": result.get("aggregate", {}),
                    "proposal_calibration": result.get("run_config", {}).get("proposal_calibration", {}),
                }
            ),
            indent=2,
        )
    )
    return 0


def _summary_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "comparison_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"comparison summary not found: {path}")
    return path


def _indices_for_split(artifact: dict, split: str) -> tuple[int, ...] | None:
    normalized = str(split or "all").lower()
    if normalized == "all":
        return None
    key = "train_indices" if normalized == "train" else "eval_indices"
    values = artifact.get(key)
    if values is None:
        raise ValueError(f"proposal calibration artifact does not contain {key}")
    return tuple(int(index) for index in values)


if __name__ == "__main__":
    raise SystemExit(main())
