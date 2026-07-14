"""Unified command dispatcher for PerceptionISP workflows."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Command:
    module: str
    summary: str


COMMANDS: dict[tuple[str, str], Command] = {
    ("isp", "run"): Command("perception_isp.core.run_cli", "Run the software ISP pipeline."),
    ("data", "coco-subset"): Command("perception_isp.datasets.prepare_coco_subset", "Prepare a small COCO validation subset."),
    ("data", "yolo-aux"): Command("perception_isp.datasets.yolo_aux_dataset", "Export a YOLO RGB+Aux dataset."),
    ("data", "resplit"): Command("perception_isp.datasets.yolo_resplit_dataset", "Create leakage-safe train/validation/test splits."),
    ("data", "lis-rgb-aux"): Command("perception_isp.datasets.lis_rgb_aux_seg_dataset", "Build a LIS RGB+Aux segmentation dataset."),
    ("data", "lis-yolo"): Command("perception_isp.datasets.lis_yolo_seg_dataset", "Convert LIS masks to YOLO segmentation."),
    ("data", "aodraw-acquire"): Command("perception_isp.datasets.aodraw_acquisition", "Run guarded AODRaw acquisition."),
    ("data", "aodraw-availability"): Command("perception_isp.datasets.aodraw_image_availability", "Audit available AODRaw images."),
    ("data", "aodraw-cleanup"): Command("perception_isp.datasets.aodraw_storage_cleanup", "Plan or execute AODRaw cleanup."),
    ("data", "aodraw-pipeline"): Command("perception_isp.datasets.aodraw_pipeline", "Run the AODRaw import and evaluation pipeline."),
    ("data", "aodraw-watch"): Command("perception_isp.datasets.aodraw_download_watch", "Watch and import AODRaw downloads."),
    ("aux", "export"): Command("perception_isp.datasets.aux_export", "Export RGB and auxiliary tensors."),
    ("train", "smoke"): Command("perception_isp.training.aux_train_smoke", "Run the compact training smoke test."),
    ("train", "dense"): Command("perception_isp.training.aux_train_dense", "Train the compact dense detector."),
    ("train", "yolo-aux"): Command("perception_isp.training.yolo_aux_train", "Fine-tune YOLO with gated early RGB+Aux fusion."),
    ("train", "feature-distill"): Command("perception_isp.training.yolo_aux_feature_distill", "Distill RGB features into the Aux stem."),
    ("train", "segmentation"): Command("perception_isp.training.scene_truth_segmentation_train", "Train the scene-truth segmentation model."),
    ("evaluate", "detection"): Command("perception_isp.evaluation.eval_cli", "Compare HumanISP and PerceptionISP detection."),
    ("evaluate", "segmentation"): Command("perception_isp.evaluation.lis_segmentation_eval", "Evaluate LIS segmentation checkpoints."),
    ("evaluate", "dense"): Command("perception_isp.evaluation.aux_eval_dense", "Evaluate compact dense checkpoints."),
    ("evaluate", "resolution"): Command("perception_isp.evaluation.resolution_sweep", "Sweep sensor output resolutions."),
    ("evaluate", "isp-sweep"): Command("perception_isp.evaluation.isp_sweep", "Sweep CFA, PSF, and ISP settings."),
    ("evaluate", "threshold"): Command("perception_isp.evaluation.threshold_sweep", "Sweep detector thresholds."),
    ("evaluate", "edge-aux"): Command("perception_isp.evaluation.scene_edge_aux_sweep", "Measure scene-edge and Aux alignment."),
    ("evaluate", "calibrate"): Command("perception_isp.evaluation.proposal_calibration", "Fit proposal calibration."),
    ("evaluate", "apply-calibration"): Command("perception_isp.evaluation.proposal_calibration_apply", "Apply proposal calibration."),
    ("report", "rollup"): Command("perception_isp.reporting.report_rollup", "Build an experiment rollup."),
    ("report", "merge"): Command("perception_isp.reporting.merge_comparison_reports", "Merge sharded comparison reports."),
    ("report", "edge-casebook"): Command("perception_isp.reporting.scene_edge_casebook", "Build an edge evidence casebook."),
    ("report", "project"): Command("perception_isp.reporting.project_accomplishment_report", "Build the project accomplishment report."),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if len(args) < 2:
        _print_group_help(args[0])
        return 2

    key = (args[0], args[1])
    command = COMMANDS.get(key)
    if command is None:
        print(f"Unknown command: {' '.join(key)}", file=sys.stderr)
        _print_group_help(args[0])
        return 2

    module = importlib.import_module(command.module)
    entrypoint: Callable[[Sequence[str] | None], int] = getattr(module, "main")
    original_program = sys.argv[0]
    sys.argv[0] = f"perception-isp {key[0]} {key[1]}"
    try:
        return int(entrypoint(args[2:]) or 0)
    finally:
        sys.argv[0] = original_program


def _print_help() -> None:
    print("usage: perception-isp <group> <command> [options]\n")
    print("Perception-oriented ISP simulation, training, and evaluation.\n")
    for group in sorted({key[0] for key in COMMANDS}):
        print(f"{group}:")
        for (candidate, name), command in sorted(COMMANDS.items()):
            if candidate == group:
                print(f"  {name:<20} {command.summary}")


def _print_group_help(group: str) -> None:
    commands = [(name, command) for (candidate, name), command in COMMANDS.items() if candidate == group]
    if not commands:
        print(f"Unknown command group: {group}", file=sys.stderr)
        _print_help()
        return
    print(f"usage: perception-isp {group} <command> [options]\n")
    for name, command in sorted(commands):
        print(f"  {name:<20} {command.summary}")


if __name__ == "__main__":
    raise SystemExit(main())
