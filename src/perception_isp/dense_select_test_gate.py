"""Selection/test-separated gate for compact RGB+aux dense detectors."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html as html_lib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .aux_eval_dense import evaluate_dense_manifest, parse_label_list
from .types import json_ready


SUMMARY_FILENAME = "summary.json"
SPLIT_FILENAME = "split_indices.json"
METRIC_KEYS = {
    "precision": "precision@0.50_mean",
    "recall": "recall@0.50_mean",
    "fp": "fp@0.50_mean",
    "det_count": "det_count_mean",
    "small_recall": "small_recall@0.50_mean",
}
DEFAULT_THRESHOLDS = (0.0, 0.02, 0.04, 0.055, 0.08, 0.10, 0.12, 0.16, 0.20)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run a selection/test-separated RGB+Aux dense-detector gate.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-source-summary", required=True, help="Training summary containing eval_indices to subdivide.")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--rgb-only-template", required=True, help="str.format template with {seed}.")
    parser.add_argument("--aux-checkpoint-template", required=True, help="str.format template with {seed} and {epoch}.")
    parser.add_argument("--epochs", default="0-11", help="Comma-separated epoch ids or ranges, for example 0-11.")
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--rgb-thresholds", default=None, help="RGB-only thresholds used when --tune-rgb-baseline is enabled; defaults to --thresholds.")
    parser.add_argument("--nms-ious", default="none", help="Comma-separated NMS IoU candidates applied to both RGB-only and RGB+Aux; use none for checkpoint default.")
    parser.add_argument("--max-detections", default="none", help="Comma-separated top-k candidates applied to both RGB-only and RGB+Aux; use none for checkpoint default.")
    parser.add_argument(
        "--tune-rgb-baseline",
        action="store_true",
        help="Also select the RGB-only operating point on the selection subset under the same FP budget.",
    )
    parser.add_argument(
        "--aux-fp-budget-source",
        default="default_rgb",
        choices=("default_rgb", "selected_rgb"),
        help="FP budget used to select RGB+Aux candidates when --tune-rgb-baseline is enabled.",
    )
    parser.add_argument("--split-seed", default="native1000-select-test-v1")
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--include-labels", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--jobs", type=int, default=1, help="Number of seeds to evaluate in parallel.")
    parser.add_argument("--output-dir", default="reports/perception_rgb_aux_dense_select_test_gate")
    args = parser.parse_args(argv)

    split_source = json.loads(Path(args.split_source_summary).expanduser().read_text())
    source_eval_indices = tuple(int(index) for index in split_source.get("eval_indices", ()))
    if not source_eval_indices:
        raise ValueError("split-source-summary must contain a non-empty eval_indices list")
    summary = build_dense_select_test_gate(
        manifest_path=args.manifest,
        source_eval_indices=source_eval_indices,
        seeds=parse_int_list(args.seeds),
        rgb_only_template=str(args.rgb_only_template),
        aux_checkpoint_template=str(args.aux_checkpoint_template),
        epochs=parse_int_list(args.epochs),
        thresholds=parse_float_list(args.thresholds),
        rgb_thresholds=parse_float_list(args.rgb_thresholds) if args.rgb_thresholds is not None else None,
        nms_ious=parse_optional_float_list(args.nms_ious),
        max_detections=parse_optional_int_list(args.max_detections),
        tune_rgb_baseline=bool(args.tune_rgb_baseline),
        aux_fp_budget_source=str(args.aux_fp_budget_source),
        split_seed=str(args.split_seed),
        selection_fraction=float(args.selection_fraction),
        include_labels=parse_label_list(args.include_labels),
        device=str(args.device),
        jobs=int(args.jobs),
        split_source_summary=str(args.split_source_summary),
    )
    html_path = write_dense_select_test_gate(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "split_json": str(html_path.parent / SPLIT_FILENAME),
                    "status": summary["status"],
                    "claim_status": summary["claim_status"],
                    "pass_test_seed_count": summary["pass_test_seed_count"],
                    "seed_count": summary["seed_count"],
                    "mean_test_deltas": summary["mean_test_deltas"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_dense_select_test_gate(
    *,
    manifest_path: str | Path,
    source_eval_indices: Sequence[int],
    seeds: Sequence[int],
    rgb_only_template: str,
    aux_checkpoint_template: str,
    epochs: Sequence[int],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    rgb_thresholds: Sequence[float] | None = None,
    nms_ious: Sequence[float | None] = (None,),
    max_detections: Sequence[int | None] = (None,),
    tune_rgb_baseline: bool = False,
    aux_fp_budget_source: str = "default_rgb",
    split_seed: str = "native1000-select-test-v1",
    selection_fraction: float = 0.5,
    include_labels: Sequence[str] | None = None,
    device: str = "auto",
    jobs: int = 1,
    split_source_summary: str | Path | None = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    selection_indices, test_indices = split_indices_by_hash(
        source_eval_indices,
        selection_fraction=selection_fraction,
        split_seed=split_seed,
    )
    seed_jobs = [
        {
            "manifest_path": manifest_path,
            "seed": int(seed),
            "rgb_only_checkpoint": rgb_only_template.format(seed=int(seed)),
            "aux_checkpoint_template": aux_checkpoint_template,
            "epochs": epochs,
            "thresholds": thresholds,
            "rgb_thresholds": rgb_thresholds if rgb_thresholds is not None else thresholds,
            "nms_ious": nms_ious,
            "max_detections": max_detections,
            "tune_rgb_baseline": bool(tune_rgb_baseline),
            "aux_fp_budget_source": str(aux_fp_budget_source),
            "selection_indices": selection_indices,
            "test_indices": test_indices,
            "include_labels": include_labels,
            "device": device,
        }
        for seed in seeds
    ]
    worker_count = max(int(jobs), 1)
    if worker_count > 1 and len(seed_jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(seed_jobs))) as executor:
            rows = list(executor.map(_run_seed_gate_from_kwargs, seed_jobs))
    else:
        rows = [
            _run_seed_gate_from_kwargs(seed_job)
            for seed_job in seed_jobs
        ]
    mean_selection_deltas = _mean_deltas(rows, "selection_deltas")
    mean_test_deltas = _mean_deltas(rows, "test_deltas")
    pass_test = [row for row in rows if row.get("test_status") == "pass"]
    status = "pass" if len(pass_test) == len(rows) else "mixed"
    return {
        "name": "Selection/test-separated RGB+Aux dense detector gate",
        "status": status,
        "claim_status": _claim_status(bool(tune_rgb_baseline), status, aux_fp_budget_source=str(aux_fp_budget_source)),
        "manifest": str(manifest_path),
        "split_source_summary": None if split_source_summary is None else str(split_source_summary),
        "split_method": f"sha1({split_seed}:index), first fraction selection, remainder test",
        "split_seed": str(split_seed),
        "selection_fraction": float(selection_fraction),
        "source_eval_count": int(len(source_eval_indices)),
        "selection_sample_count": int(len(selection_indices)),
        "test_sample_count": int(len(test_indices)),
        "selection_indices": [int(index) for index in selection_indices],
        "test_indices": [int(index) for index in test_indices],
        "seeds": [int(seed) for seed in seeds],
        "seed_count": int(len(rows)),
        "pass_test_seed_count": int(len(pass_test)),
        "candidate_thresholds": [float(value) for value in thresholds],
        "candidate_rgb_thresholds": [float(value) for value in (rgb_thresholds if rgb_thresholds is not None else thresholds)],
        "candidate_nms_ious": [None if value is None else float(value) for value in nms_ious],
        "candidate_max_detections": [None if value is None else int(value) for value in max_detections],
        "candidate_epochs": [int(value) for value in epochs],
        "tune_rgb_baseline": bool(tune_rgb_baseline),
        "aux_fp_budget_source": str(aux_fp_budget_source),
        "jobs": int(worker_count),
        "selection_rule": (
            (
                "tune RGB-only threshold and RGB+Aux epoch/threshold on the selection subset under the same "
                f"FP budget source ({aux_fp_budget_source}); maximize recall, then precision"
            )
            if bool(tune_rgb_baseline)
            else (
                "choose a candidate on the selection subset with non-increased FP, "
                "non-decreased precision, and non-decreased recall; maximize recall delta, then precision delta"
            )
        ),
        "mean_selection_deltas": mean_selection_deltas,
        "mean_test_deltas": mean_test_deltas,
        "rows": rows,
        "elapsed_seconds": float(max(time.perf_counter() - start, 0.0)),
        "important_limitations": [
            "This gate evaluates a compact dense detector, not a production detector.",
            "The final test subset is held out from metric selection, but it is still a small subset.",
            "The underlying training may have logged eval loss on the original eval split; metric selection here only uses the selection subset.",
        ],
    }


def _run_seed_gate_from_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    return _run_seed_gate(**dict(kwargs))


def split_indices_by_hash(
    indices: Sequence[int],
    *,
    selection_fraction: float,
    split_seed: str,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    values = tuple(int(index) for index in indices)
    if len(values) < 2:
        raise ValueError("at least two source indices are required for selection/test split")
    fraction = min(max(float(selection_fraction), 0.0), 1.0)
    selection_count = min(max(int(len(values) * fraction), 1), len(values) - 1)
    ordered = sorted(values, key=lambda index: hashlib.sha1(f"{split_seed}:{index}".encode("utf-8")).hexdigest())
    return tuple(sorted(ordered[:selection_count])), tuple(sorted(ordered[selection_count:]))


def write_dense_select_test_gate(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "split_method": summary.get("split_method"),
        "split_seed": summary.get("split_seed"),
        "source_eval_count": summary.get("source_eval_count"),
        "selection_sample_count": summary.get("selection_sample_count"),
        "test_sample_count": summary.get("test_sample_count"),
        "selection_indices": summary.get("selection_indices", ()),
        "test_indices": summary.get("test_indices", ()),
    }
    (destination / SPLIT_FILENAME).write_text(json.dumps(json_ready(split_payload), indent=2) + "\n")
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _claim_status(tune_rgb_baseline: bool, status: str, *, aux_fp_budget_source: str = "default_rgb") -> str:
    if bool(tune_rgb_baseline):
        if str(aux_fp_budget_source) == "selected_rgb":
            return "strict_fair_tuned_rgb_baseline_heldout_pass" if status == "pass" else "strict_fair_tuned_rgb_baseline_mixed"
        return "fair_tuned_rgb_baseline_heldout_pass" if status == "pass" else "fair_tuned_rgb_baseline_mixed"
    return "heldout_test_equal_fp_3seed_improvement" if status == "pass" else "heldout_test_mixed_result"


def parse_int_list(value: str) -> Tuple[int, ...]:
    values = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = [int(part.strip()) for part in token.split("-", 1)]
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    return tuple(dict.fromkeys(values))


def parse_float_list(value: str) -> Tuple[float, ...]:
    values = []
    for token in str(value or "").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return tuple(values)


def parse_optional_float_list(value: str) -> Tuple[float | None, ...]:
    values: list[float | None] = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "null", "default"}:
            values.append(None)
        else:
            values.append(float(token))
    return tuple(values) or (None,)


def parse_optional_int_list(value: str) -> Tuple[int | None, ...]:
    values: list[int | None] = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "null", "default"}:
            values.append(None)
        else:
            values.append(int(token))
    return tuple(values) or (None,)


def _run_seed_gate(
    *,
    manifest_path: str | Path,
    seed: int,
    rgb_only_checkpoint: str,
    aux_checkpoint_template: str,
    epochs: Sequence[int],
    thresholds: Sequence[float],
    rgb_thresholds: Sequence[float],
    nms_ious: Sequence[float | None],
    max_detections: Sequence[int | None],
    tune_rgb_baseline: bool,
    aux_fp_budget_source: str,
    selection_indices: Sequence[int],
    test_indices: Sequence[int],
    include_labels: Sequence[str] | None,
    device: str,
) -> Dict[str, Any]:
    rgb_selection = _eval_metrics(
        manifest_path=manifest_path,
        checkpoint=rgb_only_checkpoint,
        confidence=0.0,
        indices=selection_indices,
        include_labels=include_labels,
        device=device,
    )
    rgb_candidates = []
    for confidence in rgb_thresholds:
        for nms_iou in nms_ious:
            for max_detection_count in max_detections:
                rgb_candidates.append(
                    {
                        "seed": int(seed),
                        "model": "rgb_only",
                        "confidence": float(confidence),
                        "nms_iou": None if nms_iou is None else float(nms_iou),
                        "max_detections": None if max_detection_count is None else int(max_detection_count),
                        "checkpoint": str(rgb_only_checkpoint),
                        "metrics": _eval_metrics(
                            manifest_path=manifest_path,
                            checkpoint=rgb_only_checkpoint,
                            confidence=float(confidence),
                            nms_iou=nms_iou,
                            max_detections=max_detection_count,
                            indices=selection_indices,
                            include_labels=include_labels,
                            device=device,
                        ),
                    }
                )
    if not rgb_candidates:
        rgb_candidates = [
            {
                "seed": int(seed),
                "model": "rgb_only",
                "confidence": 0.0,
                "nms_iou": None,
                "max_detections": None,
                "checkpoint": str(rgb_only_checkpoint),
                "metrics": rgb_selection,
            }
        ]
    default_rgb_fp_budget = float(rgb_selection["fp"])
    selected_rgb = _best_under_fp_budget(rgb_candidates, default_rgb_fp_budget) if bool(tune_rgb_baseline) else {
        "seed": int(seed),
        "model": "rgb_only",
        "confidence": 0.0,
        "nms_iou": None,
        "max_detections": None,
        "checkpoint": str(rgb_only_checkpoint),
        "metrics": rgb_selection,
    }
    aux_fp_budget = float(selected_rgb["metrics"]["fp"]) if str(aux_fp_budget_source) == "selected_rgb" else default_rgb_fp_budget
    candidates = []
    for epoch in epochs:
        checkpoint = aux_checkpoint_template.format(seed=int(seed), epoch=int(epoch))
        for confidence in thresholds:
            for nms_iou in nms_ious:
                for max_detection_count in max_detections:
                    metrics = _eval_metrics(
                        manifest_path=manifest_path,
                        checkpoint=checkpoint,
                        confidence=float(confidence),
                        nms_iou=nms_iou,
                        max_detections=max_detection_count,
                        indices=selection_indices,
                        include_labels=include_labels,
                        device=device,
                    )
                    candidates.append(
                        {
                            "seed": int(seed),
                            "epoch": int(epoch),
                            "confidence": float(confidence),
                            "nms_iou": None if nms_iou is None else float(nms_iou),
                            "max_detections": None if max_detection_count is None else int(max_detection_count),
                            "checkpoint": str(checkpoint),
                            "metrics": metrics,
                            "deltas": _metric_deltas(metrics, selected_rgb["metrics"] if bool(tune_rgb_baseline) else rgb_selection),
                        }
                    )
    if bool(tune_rgb_baseline):
        selected = _best_under_fp_budget(candidates, aux_fp_budget)
        aux_within_fp_budget = float(selected["metrics"]["fp"]) <= float(aux_fp_budget) + 1.0e-9
        selection_status = "pass" if aux_within_fp_budget and float(selected["metrics"]["recall"]) >= float(selected_rgb["metrics"]["recall"]) else "no_strict_selection_pass"
    else:
        selected, selection_status = select_candidate(candidates)
        aux_within_fp_budget = float(selected["metrics"]["fp"]) <= float(aux_fp_budget) + 1.0e-9
    rgb_test = _eval_metrics(
        manifest_path=manifest_path,
        checkpoint=selected_rgb["checkpoint"],
        confidence=float(selected_rgb["confidence"]),
        nms_iou=selected_rgb.get("nms_iou"),
        max_detections=selected_rgb.get("max_detections"),
        indices=test_indices,
        include_labels=include_labels,
        device=device,
    )
    aux_test = _eval_metrics(
        manifest_path=manifest_path,
        checkpoint=selected["checkpoint"],
        confidence=float(selected["confidence"]),
        nms_iou=selected.get("nms_iou"),
        max_detections=selected.get("max_detections"),
        indices=test_indices,
        include_labels=include_labels,
        device=device,
    )
    test_deltas = _metric_deltas(aux_test, rgb_test)
    return {
        "seed": int(seed),
        "selection_status": selection_status,
        "selected_epoch": int(selected["epoch"]),
        "selected_confidence": float(selected["confidence"]),
        "selected_nms_iou": selected.get("nms_iou"),
        "selected_max_detections": selected.get("max_detections"),
        "selected_rgb_confidence": float(selected_rgb["confidence"]),
        "selected_rgb_nms_iou": selected_rgb.get("nms_iou"),
        "selected_rgb_max_detections": selected_rgb.get("max_detections"),
        "selection_default_rgb_fp_budget": default_rgb_fp_budget,
        "selection_fp_budget": aux_fp_budget,
        "selection_aux_within_fp_budget": bool(aux_within_fp_budget),
        "aux_fp_budget_source": str(aux_fp_budget_source),
        "tune_rgb_baseline": bool(tune_rgb_baseline),
        "rgb_checkpoint": str(rgb_only_checkpoint),
        "aux_checkpoint": str(selected["checkpoint"]),
        "selection_rgb_only_default": rgb_selection,
        "selection_rgb_only": selected_rgb["metrics"],
        "selection_rgb_aux": selected["metrics"],
        "selection_deltas": selected["deltas"],
        "selection_candidate_count": int(len(candidates)),
        "selection_rgb_candidate_count": int(len(rgb_candidates)),
        "selection_valid_count": int(_valid_candidate_count(candidates)),
        "test_rgb_only": rgb_test,
        "test_rgb_aux": aux_test,
        "test_deltas": test_deltas,
        "test_status": "pass" if _candidate_passes(test_deltas) else "mixed",
    }


def select_candidate(candidates: Sequence[Mapping[str, Any]]) -> Tuple[Mapping[str, Any], str]:
    valid = [candidate for candidate in candidates if _candidate_passes(candidate.get("deltas", {}))]
    if valid:
        return max(valid, key=_candidate_sort_key), "pass"
    if not candidates:
        raise ValueError("no candidates provided")
    return max(candidates, key=_fallback_candidate_sort_key), "no_equal_fp_candidate"


def _best_under_fp_budget(candidates: Sequence[Mapping[str, Any]], fp_budget: float) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("no candidates provided")
    valid = [candidate for candidate in candidates if float(candidate.get("metrics", {}).get("fp", 0.0)) <= float(fp_budget) + 1.0e-9]
    if not valid:
        valid = list(candidates)
    return max(
        valid,
        key=lambda candidate: (
            float(candidate.get("metrics", {}).get("recall", 0.0)),
            float(candidate.get("metrics", {}).get("precision", 0.0)),
            -float(candidate.get("metrics", {}).get("fp", 0.0)),
        ),
    )


def _eval_metrics(
    *,
    manifest_path: str | Path,
    checkpoint: str | Path,
    confidence: float,
    nms_iou: float | None = None,
    max_detections: int | None = None,
    indices: Sequence[int],
    include_labels: Sequence[str] | None,
    device: str,
) -> Dict[str, float]:
    summary = evaluate_dense_manifest(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint,
        split="eval",
        confidence=float(confidence),
        nms_iou=None if nms_iou is None else float(nms_iou),
        max_detections=None if max_detections is None else int(max_detections),
        device=device,
        label_agnostic=False,
        include_labels=include_labels,
        indices=indices,
        output_dir=None,
    )
    aggregate = summary.get("aggregate", {})
    return {name: float(aggregate.get(key, 0.0)) for name, key in METRIC_KEYS.items()}


def _metric_deltas(metrics: Mapping[str, float], baseline: Mapping[str, float]) -> Dict[str, float]:
    return {key: float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in METRIC_KEYS}


def _candidate_passes(deltas: Mapping[str, Any]) -> bool:
    return (
        float(deltas.get("fp", 0.0)) <= 1.0e-9
        and float(deltas.get("precision", 0.0)) >= -1.0e-12
        and float(deltas.get("recall", 0.0)) >= -1.0e-12
    )


def _valid_candidate_count(candidates: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for candidate in candidates if _candidate_passes(candidate.get("deltas", {})))


def _candidate_sort_key(candidate: Mapping[str, Any]) -> Tuple[float, float, float]:
    deltas = candidate.get("deltas", {})
    metrics = candidate.get("metrics", {})
    return (
        float(deltas.get("recall", 0.0)),
        float(deltas.get("precision", 0.0)),
        -float(metrics.get("fp", 0.0)),
    )


def _fallback_candidate_sort_key(candidate: Mapping[str, Any]) -> Tuple[float, float, float]:
    deltas = candidate.get("deltas", {})
    return (
        float(deltas.get("recall", 0.0)),
        float(deltas.get("precision", 0.0)),
        -max(float(deltas.get("fp", 0.0)), 0.0),
    )


def _mean_deltas(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    if not rows:
        return {name: 0.0 for name in METRIC_KEYS}
    return {
        metric: float(statistics.mean(float(row.get(key, {}).get(metric, 0.0)) for row in rows))
        for metric in METRIC_KEYS
    }


def _render_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for row in summary.get("rows", ()):
        deltas = row.get("test_deltas", {})
        rgb = row.get("test_rgb_only", {})
        aux = row.get("test_rgb_aux", {})
        rows.append(
            "<tr>"
            f"<td>{int(row.get('seed', 0))}</td>"
            f"<td>{_fmt(row.get('selected_rgb_confidence'), digits=3)}</td>"
            f"<td>{int(row.get('selected_epoch', 0))}</td>"
            f"<td>{_fmt(row.get('selected_confidence'), digits=3)}</td>"
            f"<td>{_fmt(rgb.get('precision'))}</td><td>{_fmt(aux.get('precision'))}</td><td class=\"{_delta_class(deltas.get('precision'), positive_good=True)}\">{_fmt(deltas.get('precision'), signed=True)}</td>"
            f"<td>{_fmt(rgb.get('recall'))}</td><td>{_fmt(aux.get('recall'))}</td><td class=\"{_delta_class(deltas.get('recall'), positive_good=True)}\">{_fmt(deltas.get('recall'), signed=True)}</td>"
            f"<td>{_fmt(rgb.get('fp'), digits=3)}</td><td>{_fmt(aux.get('fp'), digits=3)}</td><td class=\"{_delta_class(deltas.get('fp'), positive_good=False)}\">{_fmt(deltas.get('fp'), signed=True, digits=3)}</td>"
            f"<td>{html_lib.escape(str(row.get('test_status', '')))}</td>"
            "</tr>"
        )
    mean_test = summary.get("mean_test_deltas", {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PerceptionISP Selection/Test RGB+Aux Gate</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; color: #17202a; background: #fff; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 44px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 20px; margin-top: 28px; }}
    p, li {{ color: #5f6b7a; line-height: 1.5; }}
    .status {{ display: inline-block; padding: 4px 10px; border: 1px solid #b8e0cd; background: #ecf8f2; color: #0b7a4b; border-radius: 6px; font-weight: 700; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }}
    section {{ border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; }}
    .big {{ font-size: 28px; font-weight: 800; margin: 4px 0; color: #17202a; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d8dee8; padding: 9px 8px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f5f7fa; }}
    .pos {{ color: #0b7a4b; font-weight: 700; }}
    .neg {{ color: #b42318; font-weight: 700; }}
    .note {{ border-left: 4px solid #9a6200; background: #fff8ea; padding: 12px 14px; color: #513b08; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 620px) {{ .cards {{ grid-template-columns: 1fr; }} main {{ padding: 20px 14px; }} }}
  </style>
</head>
<body>
<main>
  <h1>PerceptionISP Selection/Test RGB+Aux Gate</h1>
  <p class="status">{html_lib.escape(str(summary.get('claim_status', '')))}</p>
  <p>Operating points are selected on the selection subset and then applied unchanged to the held-out test subset.</p>
  <div class="cards">
    <section><h3>Test Pass Seeds</h3><p class="big">{int(summary.get('pass_test_seed_count', 0))}/{int(summary.get('seed_count', 0))}</p></section>
    <section><h3>Mean Test Recall Delta</h3><p class="big {_delta_class(mean_test.get('recall'), positive_good=True)}">{_fmt(mean_test.get('recall'), signed=True)}</p></section>
    <section><h3>Mean Test FP Delta</h3><p class="big {_delta_class(mean_test.get('fp'), positive_good=False)}">{_fmt(mean_test.get('fp'), signed=True, digits=3)}</p></section>
    <section><h3>Mean Test Precision Delta</h3><p class="big {_delta_class(mean_test.get('precision'), positive_good=True)}">{_fmt(mean_test.get('precision'), signed=True)}</p></section>
  </div>
  <h2>Held-Out Test Results</h2>
  <table>
    <thead><tr><th>Seed</th><th>RGB Conf.</th><th>Aux Epoch</th><th>Aux Conf.</th><th>RGB P</th><th>Aux P</th><th>Delta P</th><th>RGB R</th><th>Aux R</th><th>Delta R</th><th>RGB FP</th><th>Aux FP</th><th>Delta FP</th><th>Status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Protocol</h2>
  <ul>
    <li>Source eval samples: {int(summary.get('source_eval_count', 0))}; selection: {int(summary.get('selection_sample_count', 0))}; test: {int(summary.get('test_sample_count', 0))}.</li>
    <li>Candidate epochs: <code>{html_lib.escape(str(summary.get('candidate_epochs', [])))}</code></li>
    <li>Candidate thresholds: <code>{html_lib.escape(str(summary.get('candidate_thresholds', [])))}</code></li>
    <li>Candidate NMS IoUs: <code>{html_lib.escape(str(summary.get('candidate_nms_ious', [])))}</code>; max detections: <code>{html_lib.escape(str(summary.get('candidate_max_detections', [])))}</code></li>
    <li>Tuned RGB baseline: <code>{html_lib.escape(str(summary.get('tune_rgb_baseline', False)))}</code>; RGB thresholds: <code>{html_lib.escape(str(summary.get('candidate_rgb_thresholds', [])))}</code></li>
    <li>Aux FP budget source: <code>{html_lib.escape(str(summary.get('aux_fp_budget_source', 'default_rgb')))}</code></li>
  </ul>
  <p class="note"><strong>Boundary:</strong> this is stronger than same-split metric selection, but still compact-detector feasibility evidence rather than production detector proof.</p>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code>; split JSON: <code>{SPLIT_FILENAME}</code></p>
</main>
</body>
</html>
"""


def _fmt(value: Any, *, signed: bool = False, digits: int = 4) -> str:
    if value is None:
        return ""
    number = float(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _delta_class(value: Any, *, positive_good: bool) -> str:
    number = float(value or 0.0)
    passed = number >= -1.0e-12 if positive_good else number <= 1.0e-9
    return "pos" if passed else "neg"


if __name__ == "__main__":
    raise SystemExit(main())
