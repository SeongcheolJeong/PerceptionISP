"""Apply a saved proposal calibration model to a comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .comparison import write_comparison_report
from .proposal_calibration import apply_proposal_calibration_to_report
from .report_rollup import build_rollup, write_rollup_report
from .types import json_ready

PRINT_METRIC_KEYS = ("precision@0.50_mean", "recall@0.50_mean", "recall@0.75_mean", "small_recall@0.50_mean", "fp@0.50_mean", "det_count_mean")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Apply proposal calibration model artifact to a saved comparison report.")
    parser.add_argument("report", help="Report directory or comparison_summary.json")
    parser.add_argument("--model", action="append", required=True, help="proposal_calibration_model.json. Repeat to apply multiple artifacts.")
    parser.add_argument("--split", default="all", choices=["all", "train", "eval"], help="Apply to all samples or artifact train/eval indices.")
    parser.add_argument("--output-input", default=None, help="Override output input name.")
    parser.add_argument("--output-dir", default="reports/perception_proposal_calibration_applied")
    parser.add_argument("--rollup-output-dir", default=None, help="Optional output directory for a rollup over all applied reports.")
    parser.add_argument("--include-source-report-in-rollup", action="store_true", help="Include the original report in the optional rollup.")
    parser.add_argument("--print-full-aggregate", action="store_true", help="Print the full aggregate metric payload instead of compact metrics.")
    args = parser.parse_args(argv)

    report_path = _summary_path(args.report)
    report = json.loads(report_path.read_text())
    model_paths = tuple(Path(value).expanduser() for value in args.model)
    applied_reports = []
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"proposal calibration model not found: {model_path}")
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
        output_dir = _output_dir_for_model(args.output_dir, model_path, multi_model=len(model_paths) > 1)
        html_path = write_comparison_report(result, output_dir)
        applied_reports.append(
            {
                "model": str(model_path),
                "report": str(html_path),
                "summary_json": str(html_path.parent / "comparison_summary.json"),
                "aggregate": _printable_aggregate(result.get("aggregate", {}), full=bool(args.print_full_aggregate)),
                "proposal_calibration": result.get("run_config", {}).get("proposal_calibration", {}),
            }
        )

    rollup_path = None
    if args.rollup_output_dir:
        rollup_inputs: Sequence[str | Path] = tuple(
            ([report_path] if bool(args.include_source_report_in_rollup) else [])
            + [item["summary_json"] for item in applied_reports]
        )
        rollup = build_rollup(rollup_inputs)
        rollup_path = write_rollup_report(rollup, args.rollup_output_dir)

    first = applied_reports[0] if applied_reports else {}
    print(
        json.dumps(
            json_ready(
                {
                    "report": first.get("report"),
                    "summary_json": first.get("summary_json"),
                    "report_count": int(len(applied_reports)),
                    "reports": applied_reports,
                    "rollup": str(rollup_path) if rollup_path else None,
                    "aggregate": first.get("aggregate", {}),
                    "proposal_calibration": first.get("proposal_calibration", {}),
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


def _printable_aggregate(aggregate: Any, *, full: bool) -> Any:
    if full or not isinstance(aggregate, dict):
        return aggregate
    compact = {}
    for input_name, metrics in aggregate.items():
        if not isinstance(metrics, dict):
            continue
        compact[str(input_name)] = {key: metrics.get(key) for key in PRINT_METRIC_KEYS if key in metrics}
    return compact


def _output_dir_for_model(output_dir: str | Path, model_path: Path, *, multi_model: bool) -> Path:
    base = Path(output_dir).expanduser()
    if not multi_model:
        return base
    stem = model_path.stem
    prefix = "proposal_calibration_model_"
    if stem == "proposal_calibration_model":
        name = "default"
    elif stem.startswith(prefix):
        name = stem[len(prefix) :]
    else:
        name = stem
    return base / _safe_name(name)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))


if __name__ == "__main__":
    raise SystemExit(main())
