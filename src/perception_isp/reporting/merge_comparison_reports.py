"""Merge disjoint comparison report shards into one comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from perception_isp.evaluation.comparison import _aggregate_breakdown, _input_names_from_samples, write_comparison_report
from perception_isp.evaluation.metrics import aggregate_metric_rows
from perception_isp.core.types import json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Merge comparison_summary.json shards into one report.")
    parser.add_argument("reports", nargs="+", help="comparison_summary.json paths or report directories.")
    parser.add_argument("--output-dir", default="reports/perception_compare_merged")
    parser.add_argument("--name", default=None, help="Optional merged run name stored in run_config.")
    args = parser.parse_args(argv)

    merged = merge_comparison_reports(args.reports, name=args.name)
    html_path = write_comparison_report(merged, args.output_dir)
    print(json.dumps(json_ready({"report": str(html_path), "sample_count": merged["sample_count"]}), indent=2))
    return 0


def merge_comparison_reports(paths: Sequence[str | Path], *, name: str | None = None) -> Dict[str, Any]:
    summaries = [_load_summary(path) for path in paths]
    samples = [sample for summary in summaries for sample in summary.get("samples", ())]
    aggregate: Dict[str, Any] = {}
    for input_name in _input_names_from_samples(samples):
        rows = [sample["metrics"][input_name] for sample in samples if input_name in sample.get("metrics", {})]
        aggregate[input_name] = aggregate_metric_rows(rows)
    run_configs = [summary.get("run_config", {}) for summary in summaries]
    merged = {
        "sample_count": int(len(samples)),
        "aggregate": aggregate,
        "breakdown": _aggregate_breakdown(samples),
        "samples": samples,
        "run_config": _merged_run_config(run_configs, summaries, name=name),
    }
    return merged


def _load_summary(path: str | Path) -> Mapping[str, Any]:
    summary_path = _summary_path(path)
    return json.loads(summary_path.read_text())


def _summary_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "comparison_summary.json"
    if not candidate.exists():
        raise FileNotFoundError(f"comparison summary not found: {candidate}")
    return candidate


def _merged_run_config(
    run_configs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    *,
    name: str | None,
) -> Dict[str, Any]:
    base = dict(run_configs[0]) if run_configs else {}
    total = sum(int(summary.get("sample_count", 0)) for summary in summaries)
    offsets = [config.get("offset") for config in run_configs if config.get("offset") is not None]
    counts = [config.get("count") for config in run_configs if config.get("count") is not None]
    base["merged"] = True
    base["merged_name"] = name
    base["merged_shard_count"] = int(len(summaries))
    base["merged_sample_count"] = int(total)
    base["merged_offsets"] = offsets
    base["merged_counts"] = counts
    if offsets:
        base["offset"] = min(int(value) for value in offsets)
    base["count"] = int(total)
    return base


if __name__ == "__main__":
    raise SystemExit(main())
