"""Build a standalone tabbed report focused on PerceptionISP achievements."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_PATHS = {
    "native_gate": Path(
        "reports/perception_native_heldout_benchmark_audit_pascalraw_rawdet7_val750_fusion_v2/"
        "native_heldout_benchmark_audit_summary.json"
    ),
    "dense_ablation": Path(
        "reports/perception_dense_input_ablation_gate_pascalraw_native1581_union_seed101_202_303_broad_cached_v1/"
        "dense_input_ablation_summary.json"
    ),
    "scene_edge": Path("reports/perception_scene_edge_confidence_bus_cfa_psf_sweep/scene_edge_confidence_summary.json"),
    "scene_edge_aux": Path(
        "reports/perception_scene_edge_aux_sweep_sid_lowlight_native16_edge_evidence_v1/"
        "scene_edge_aux_sweep_summary.json"
    ),
    "yolo_audit": Path("reports/perception_yolo_edge6_evidence_audit_current_v1/yolo_evidence_audit_summary.json"),
    "raw_acquisition": Path("reports/perception_raw_dataset_acquisition_lod_priority_v1/raw_dataset_acquisition_summary.json"),
    "lod_readiness": Path("reports/perception_lod_local_readiness_current_v1/lod_local_readiness_summary.json"),
    "hard_seed2": Path("reports/perception_yolo_hard_object_eval_seed2_edge6_vs_rgb_pascalraw1581_v1/summary.json"),
    "hard_gated_seed2": Path(
        "reports/perception_yolo_hard_object_eval_seed2_gated_edge6_vs_rgb_pascalraw1581_v1/summary.json"
    ),
    "hard_hardx3_seed2": Path(
        "reports/perception_yolo_hard_object_eval_seed2_hardx3_ft_rgb_vs_gated_pascalraw1581_v1/summary.json"
    ),
    "hard_freezergb_seed2": Path(
        "reports/perception_yolo_hard_object_eval_seed2_rgb_hardx3_to_aux_freezergb_pascalraw1581_v1/"
        "summary.json"
    ),
    "hard_freezergb_vs_rgb_continue_seed2": Path(
        "reports/perception_yolo_hard_object_eval_seed2_rgb_continue_vs_aux_freezergb_pascalraw1581_v1/"
        "summary.json"
    ),
    "hard_hardx5_seed2": Path(
        "reports/perception_yolo_hard_object_eval_seed2_hardx5_rgb_vs_aux_pascalraw1581_v1/summary.json"
    ),
    "hard_fixedopt_gate3_seed2": Path(
        "reports/perception_yolo_hard_object_eval_seed2_rgb_continue_vs_aux_fixedopt_gate3_pascalraw1581_v1/"
        "summary.json"
    ),
    "hard_fixedopt_gate1_seed2": Path(
        "reports/perception_yolo_hard_object_eval_seed2_rgb_continue_vs_aux_gate1_screen_pascalraw1581_v1/"
        "summary.json"
    ),
    "hard_fixedopt_gate1_e4_seed2": Path(
        "reports/perception_yolo_hard_object_eval_seed2_rgb_continue_vs_aux_gate1_e4_pascalraw1581_v1/"
        "summary.json"
    ),
    "hard_fixedopt_gate1_e2_seed3": Path(
        "reports/perception_yolo_hard_object_eval_seed3_rgb_continue_e2_vs_aux_gate1_e2_pascalraw1581_v1/"
        "summary.json"
    ),
    "hard_fixedopt_gate1_e2_seed4": Path(
        "reports/perception_yolo_hard_object_eval_seed4_rgb_continue_e2_vs_aux_gate1_e2_pascalraw1581_v1/"
        "summary.json"
    ),
    "hard_candidates": Path("reports/perception_yolo_hard_object_eval_gated_candidates_pascalraw1581_v1/summary.json"),
    "resplit_rgb_seed101": Path("exports/perception_yolo_rgb_only_pascalraw_native1581_resplit_seed101_hardx3_v1/summary.json"),
    "resplit_aux_seed101": Path("exports/perception_yolo_rgb_aux_edge6_pascalraw_native1581_resplit_seed101_hardx3_v1/summary.json"),
}

TRAIN_SUMMARIES = {
    "rgb_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_only3_full_e20_mps_seed2_adamw_lr5e4_v1/perception_yolo_aux_train_summary.json"
    ),
    "edge6_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_aux_edge6_adapter025_full_e20_mps_seed2_adamw_lr5e4_v1/perception_yolo_aux_train_summary.json"
    ),
    "gated_edge6_seed2_matched": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_aux_edge6_gated_adapter025_full_e20_mps_seed2_matched_adamw_lr5e4_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "rgb_hardx3_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_only_full_hardx3_ft_e4_mps_seed2_adamw_lr2e4_b8_matched_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "gated_hardx3_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_aux_gated_full_hardx3_ft_e4_mps_seed2_adamw_lr2e4_b8_matched_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "rgb_hardx3_to_aux_freezergb_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_to_aux_gated_freezergb_adapter025_gate3_e4_mps_seed2_adamw_lr1e4_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "rgb_hardx3_continue_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_continue_e4_mps_seed2_adamw_lr1e4_b8_control_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "rgb_hardx5_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_from_hardx3_to_hardx5_e3_mps_seed2_adamw_lr8e5_b8_control_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "aux_freezergb_hardx5_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "aux_freezergb_from_hardx3_to_hardx5_e3_mps_seed2_adamw_lr8e5_b8_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "aux_fixedopt_gate3_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_to_aux_gated_freezergb_adapter025_gate3_e4_mps_seed2_adamw_lr1e4_fixedopt_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "aux_fixedopt_gate1_screen_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_to_aux_gated_freezergb_adapter025_gate1_e2_mps_seed2_adamw_lr1e4_fixedopt_screen_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "aux_fixedopt_gate1_e4_seed2": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_to_aux_gated_freezergb_adapter025_gate1_e4_mps_seed2_adamw_lr1e4_fixedopt_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "rgb_continue_e2_seed3": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_continue_e2_mps_seed3_adamw_lr1e4_b8_control_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "aux_fixedopt_gate1_e2_seed3": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_to_aux_gated_freezergb_adapter025_gate1_e2_mps_seed3_adamw_lr1e4_fixedopt_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "rgb_continue_e2_seed4": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_continue_e2_mps_seed4_adamw_lr1e4_b8_control_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "aux_fixedopt_gate1_e2_seed4": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_hardx3_to_aux_gated_freezergb_adapter025_gate1_e2_mps_seed4_adamw_lr1e4_fixedopt_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "aux_optimizer_fix_smoke": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "aux_stem_optimizer_state_smoke_frac005_e1_mps_seed2_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "rgb_resplit_seed101_smoke": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "rgb_resplit_seed101_hardx3_yolo11n_smoke_frac005_e1_mps_seed101_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
    "aux_resplit_seed101_smoke": Path(
        "runs/detect/outputs/yolo_aux_pascalraw1581/"
        "aux_resplit_seed101_hardx3_yolo11n_gated_meanrgb_smoke_frac005_e1_mps_seed101_v1/"
        "perception_yolo_aux_train_summary.json"
    ),
}

LOD_EXPECTED_BYTES = 22_000_000_000


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the current PerceptionISP achievement/storyline HTML report.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--output-dir", default="reports/perception_project_accomplishment_tabs_current_v1")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    report = build_report(root)
    html_path = write_report(report, root / args.output_dir)
    print(
        json.dumps(
            {
                "report": str(html_path),
                "summary_json": str(html_path.with_name("project_accomplishment_summary.json")),
            },
            indent=2,
        )
    )
    return 0


def build_report(root: Path) -> dict[str, Any]:
    summaries = {name: _read_json(root / path) for name, path in SUMMARY_PATHS.items()}
    train = {name: _read_train_summary(root, path) for name, path in TRAIN_SUMMARIES.items()}

    native = _native_metrics(summaries["native_gate"])
    dense = _dense_metrics(summaries["dense_ablation"])
    scene = _scene_metrics(summaries["scene_edge"], summaries["scene_edge_aux"])
    yolo = _yolo_train_comparison(train["rgb_seed2"], train["edge6_seed2"], "edge6_seed2 - rgb_seed2")
    gated = _yolo_train_comparison(train["rgb_seed2"], train["gated_edge6_seed2_matched"], "gated_edge6_seed2 - rgb_seed2")
    hard_seed2 = _hard_pair(summaries["hard_seed2"], "rgb_seed2", "edge6_seed2", "edge6_seed2 - rgb_seed2")
    hard_gated_seed2 = _hard_pair(
        summaries["hard_gated_seed2"], "rgb_seed2", "gated_edge6_seed2", "gated_edge6_seed2 - rgb_seed2"
    )
    hard_hardx3_seed2 = _hard_pair(
        summaries["hard_hardx3_seed2"],
        "rgb_hardx3_seed2",
        "gated_hardx3_seed2",
        "gated_hardx3_seed2 - rgb_hardx3_seed2",
    )
    hard_freezergb_seed2 = _hard_pair(
        summaries["hard_freezergb_seed2"],
        "rgb_hardx3_seed2",
        "rgb_hardx3_to_aux_freezergb_seed2",
        "rgb_hardx3_to_aux_freezergb_seed2 - rgb_hardx3_seed2",
    )
    hard_freezergb_vs_rgb_continue_seed2 = _hard_pair(
        summaries["hard_freezergb_vs_rgb_continue_seed2"],
        "rgb_hardx3_continue_seed2",
        "rgb_hardx3_to_aux_freezergb_seed2",
        "rgb_hardx3_to_aux_freezergb_seed2 - rgb_hardx3_continue_seed2",
    )
    hard_hardx5_seed2 = _hard_pair(
        summaries["hard_hardx5_seed2"],
        "rgb_hardx5_seed2",
        "aux_freezergb_hardx5_seed2",
        "aux_freezergb_hardx5_seed2 - rgb_hardx5_seed2",
    )
    hard_fixedopt_gate3_seed2 = _hard_pair(
        summaries["hard_fixedopt_gate3_seed2"],
        "rgb_continue_seed2",
        "aux_fixedopt_gate3_seed2",
        "aux_fixedopt_gate3_seed2 - rgb_continue_seed2",
    )
    hard_fixedopt_gate1_seed2 = _hard_pair(
        summaries["hard_fixedopt_gate1_seed2"],
        "rgb_continue_seed2",
        "aux_gate1_screen_seed2",
        "aux_fixedopt_gate1_screen_seed2 - rgb_continue_seed2",
    )
    hard_fixedopt_gate1_e4_seed2 = _hard_pair(
        summaries["hard_fixedopt_gate1_e4_seed2"],
        "rgb_continue_seed2",
        "aux_gate1_e4_seed2",
        "aux_fixedopt_gate1_e4_seed2 - rgb_continue_seed2",
    )
    hard_fixedopt_gate1_e2_seed3 = _hard_pair(
        summaries["hard_fixedopt_gate1_e2_seed3"],
        "rgb_continue_seed3",
        "aux_gate1_e2_seed3",
        "aux_fixedopt_gate1_e2_seed3 - rgb_continue_seed3",
    )
    hard_fixedopt_gate1_e2_seed4 = _hard_pair(
        summaries["hard_fixedopt_gate1_e2_seed4"],
        "rgb_continue_seed4",
        "aux_gate1_e2_seed4",
        "aux_fixedopt_gate1_e2_seed4 - rgb_continue_seed4",
    )
    hardx3_train = _yolo_train_comparison(
        train["rgb_hardx3_seed2"],
        train["gated_hardx3_seed2"],
        "gated_hardx3_seed2 - rgb_hardx3_seed2",
    )
    hardx3_freezergb_train = _yolo_train_comparison(
        train["rgb_hardx3_seed2"],
        train["rgb_hardx3_to_aux_freezergb_seed2"],
        "rgb_hardx3_to_aux_freezergb_seed2 - rgb_hardx3_seed2",
    )
    hardx3_continue_train = _yolo_train_comparison(
        train["rgb_hardx3_seed2"],
        train["rgb_hardx3_continue_seed2"],
        "rgb_hardx3_continue_seed2 - rgb_hardx3_seed2",
    )
    hardx3_freezergb_vs_continue_train = _yolo_train_comparison(
        train["rgb_hardx3_continue_seed2"],
        train["rgb_hardx3_to_aux_freezergb_seed2"],
        "rgb_hardx3_to_aux_freezergb_seed2 - rgb_hardx3_continue_seed2",
    )
    hardx5_train = _yolo_train_comparison(
        train["rgb_hardx5_seed2"],
        train["aux_freezergb_hardx5_seed2"],
        "aux_freezergb_hardx5_seed2 - rgb_hardx5_seed2",
    )
    fixedopt_gate3_train = _yolo_train_comparison(
        train["rgb_hardx3_continue_seed2"],
        train["aux_fixedopt_gate3_seed2"],
        "aux_fixedopt_gate3_seed2 - rgb_hardx3_continue_seed2",
    )
    fixedopt_gate1_train = _yolo_train_comparison(
        train["rgb_hardx3_continue_seed2"],
        train["aux_fixedopt_gate1_screen_seed2"],
        "aux_fixedopt_gate1_screen_seed2 - rgb_hardx3_continue_seed2",
    )
    fixedopt_gate1_e4_train = _yolo_train_comparison(
        train["rgb_hardx3_continue_seed2"],
        train["aux_fixedopt_gate1_e4_seed2"],
        "aux_fixedopt_gate1_e4_seed2 - rgb_hardx3_continue_seed2",
    )
    fixedopt_gate1_e2_seed3_train = _yolo_train_comparison(
        train["rgb_continue_e2_seed3"],
        train["aux_fixedopt_gate1_e2_seed3"],
        "aux_fixedopt_gate1_e2_seed3 - rgb_continue_e2_seed3",
    )
    fixedopt_gate1_e2_seed4_train = _yolo_train_comparison(
        train["rgb_continue_e2_seed4"],
        train["aux_fixedopt_gate1_e2_seed4"],
        "aux_fixedopt_gate1_e2_seed4 - rgb_continue_e2_seed4",
    )
    optimizer_fix_smoke = _optimizer_fix_smoke(train["aux_optimizer_fix_smoke"])
    hard_candidates = _hard_candidate_pairs(summaries["hard_candidates"])
    lod = _lod_download_status(root)
    acquisition = _raw_acquisition_state(summaries["raw_acquisition"], summaries["lod_readiness"], lod)

    return {
        "name": "PerceptionISP accomplishment and validation report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summaries": {name: str(path) for name, path in SUMMARY_PATHS.items()},
        "train_summaries": {name: str(path) for name, path in TRAIN_SUMMARIES.items()},
        "native": native,
        "dense": dense,
        "scene": scene,
        "yolo_seed2": yolo,
        "gated_seed2": gated,
        "hard_seed2": hard_seed2,
        "hard_gated_seed2": hard_gated_seed2,
        "hard_hardx3_seed2": hard_hardx3_seed2,
        "hard_freezergb_seed2": hard_freezergb_seed2,
        "hard_freezergb_vs_rgb_continue_seed2": hard_freezergb_vs_rgb_continue_seed2,
        "hard_hardx5_seed2": hard_hardx5_seed2,
        "hard_fixedopt_gate3_seed2": hard_fixedopt_gate3_seed2,
        "hard_fixedopt_gate1_seed2": hard_fixedopt_gate1_seed2,
        "hard_fixedopt_gate1_e4_seed2": hard_fixedopt_gate1_e4_seed2,
        "hard_fixedopt_gate1_e2_seed3": hard_fixedopt_gate1_e2_seed3,
        "hard_fixedopt_gate1_e2_seed4": hard_fixedopt_gate1_e2_seed4,
        "hardx3_train": hardx3_train,
        "hardx3_freezergb_train": hardx3_freezergb_train,
        "hardx3_continue_train": hardx3_continue_train,
        "hardx3_freezergb_vs_continue_train": hardx3_freezergb_vs_continue_train,
        "hardx5_train": hardx5_train,
        "fixedopt_gate3_train": fixedopt_gate3_train,
        "fixedopt_gate1_train": fixedopt_gate1_train,
        "fixedopt_gate1_e4_train": fixedopt_gate1_e4_train,
        "fixedopt_gate1_e2_seed3_train": fixedopt_gate1_e2_seed3_train,
        "fixedopt_gate1_e2_seed4_train": fixedopt_gate1_e2_seed4_train,
        "optimizer_fix_smoke": optimizer_fix_smoke,
        "hard_candidates": hard_candidates,
        "acquisition": acquisition,
        "status": _overall_status(
            native,
            dense,
            yolo,
            hard_hardx3_seed2,
            hard_hardx5_seed2,
            hard_fixedopt_gate3_seed2,
            hard_fixedopt_gate1_seed2,
            hard_fixedopt_gate1_e4_seed2,
            hard_fixedopt_gate1_e2_seed3,
            hard_fixedopt_gate1_e2_seed4,
            hardx3_freezergb_train,
            hardx3_freezergb_vs_continue_train,
            fixedopt_gate1_train,
            fixedopt_gate1_e4_train,
            fixedopt_gate1_e2_seed3_train,
            fixedopt_gate1_e2_seed4_train,
            acquisition,
        ),
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "project_accomplishment_summary.json"
    html_path = output_dir / "index.html"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(_render_html(report), encoding="utf-8")
    return html_path


def _render_html(report: Mapping[str, Any]) -> str:
    native = _map(report.get("native"))
    dense = _map(report.get("dense"))
    scene = _map(report.get("scene"))
    yolo = _map(report.get("yolo_seed2"))
    gated = _map(report.get("gated_seed2"))
    hard = _map(report.get("hard_seed2"))
    hard_gated = _map(report.get("hard_gated_seed2"))
    hard_hardx3 = _map(report.get("hard_hardx3_seed2"))
    hard_freezergb = _map(report.get("hard_freezergb_seed2"))
    hard_freezergb_vs_continue = _map(report.get("hard_freezergb_vs_rgb_continue_seed2"))
    hard_hardx5 = _map(report.get("hard_hardx5_seed2"))
    hard_fixedopt_gate3 = _map(report.get("hard_fixedopt_gate3_seed2"))
    hard_fixedopt_gate1 = _map(report.get("hard_fixedopt_gate1_seed2"))
    hard_fixedopt_gate1_e4 = _map(report.get("hard_fixedopt_gate1_e4_seed2"))
    hard_fixedopt_gate1_e2_seed3 = _map(report.get("hard_fixedopt_gate1_e2_seed3"))
    hard_fixedopt_gate1_e2_seed4 = _map(report.get("hard_fixedopt_gate1_e2_seed4"))
    hard_primary = (
        hard_fixedopt_gate1_e4
        if hard_fixedopt_gate1_e4.get("status") != "missing"
        else hard_fixedopt_gate1
        if hard_fixedopt_gate1.get("status") != "missing"
        else hard_fixedopt_gate3
        if hard_fixedopt_gate3.get("status") != "missing"
        else hard_freezergb_vs_continue
        if hard_freezergb_vs_continue.get("status") != "missing"
        else
        hard_freezergb
        if hard_freezergb.get("status") != "missing"
        else hard_hardx3
        if hard_hardx3.get("status") != "missing"
        else hard_gated
        if hard_gated.get("status") != "missing"
        else hard
    )
    hardx3_train = _map(report.get("hardx3_train"))
    hardx3_freezergb_train = _map(report.get("hardx3_freezergb_train"))
    hardx3_continue_train = _map(report.get("hardx3_continue_train"))
    hardx3_freezergb_vs_continue_train = _map(report.get("hardx3_freezergb_vs_continue_train"))
    hardx5_train = _map(report.get("hardx5_train"))
    fixedopt_gate3_train = _map(report.get("fixedopt_gate3_train"))
    fixedopt_gate1_train = _map(report.get("fixedopt_gate1_train"))
    fixedopt_gate1_e4_train = _map(report.get("fixedopt_gate1_e4_train"))
    fixedopt_gate1_e2_seed3_train = _map(report.get("fixedopt_gate1_e2_seed3_train"))
    fixedopt_gate1_e2_seed4_train = _map(report.get("fixedopt_gate1_e2_seed4_train"))
    optimizer_fix_smoke = _map(report.get("optimizer_fix_smoke"))
    candidates = _list(report.get("hard_candidates"))
    acquisition = _map(report.get("acquisition"))
    generated = _esc(report.get("generated_at"))

    cards = [
        (
            "Native RAW gate",
            f"{_fmt(native.get('sample_count'), digits=0)} samples",
            f"dP50={_fmt(native.get('precision50_delta'), signed=True)}, "
            f"dFP50={_fmt(native.get('fp50_delta'), signed=True)}, dR50={_fmt(native.get('recall50_delta'), signed=True)}",
            native.get("status", "unknown"),
        ),
        (
            "Aux input dependence",
            f"{_fmt(dense.get('seed_count'), digits=0)} seeds",
            f"zero-aux recall delta={_fmt(dense.get('zero_aux_recall_delta'), signed=True)}",
            dense.get("status", "unknown"),
        ),
        (
            "Gated YOLO seed2",
            "RGB+Aux vs RGB",
            f"dR={_fmt(gated.get('recall_delta'), signed=True)}, "
            f"dAP50={_fmt(gated.get('map50_delta'), signed=True)}, "
            f"dAP50-95={_fmt(gated.get('map5095_delta'), signed=True)}",
            gated.get("status", "unknown"),
        ),
        (
            "RGB-preserved Aux",
            "hardx3 -> Aux",
            f"dR={_fmt(hardx3_freezergb_train.get('recall_delta'), signed=True)}, "
            f"dAP50-95={_fmt(hardx3_freezergb_train.get('map5095_delta'), signed=True)}",
            hardx3_freezergb_train.get("status", "unknown"),
        ),
        (
            "Aux vs RGB control",
            "extra epoch fair",
            f"dR={_fmt(hardx3_freezergb_vs_continue_train.get('recall_delta'), signed=True)}, "
            f"dAP50-95={_fmt(hardx3_freezergb_vs_continue_train.get('map5095_delta'), signed=True)}",
            hardx3_freezergb_vs_continue_train.get("status", "unknown"),
        ),
        (
            "Hard-object slice",
            "small/thin",
            _hard_card_text(hard_primary),
            hard_primary.get("status", "unknown"),
        ),
        (
            "Hardx5 stress",
            "oversampling check",
            _hard_card_text(hard_hardx5),
            hard_hardx5.get("status", "unknown"),
        ),
        (
            "Corrected Aux",
            "seed3/4 repeat",
            f"dP={_fmt(fixedopt_gate1_e2_seed4_train.get('precision_delta'), signed=True)}, "
            f"dAP50-95={_fmt(fixedopt_gate1_e2_seed4_train.get('map5095_delta'), signed=True)}",
            fixedopt_gate1_e2_seed4_train.get("status", "unknown"),
        ),
        (
            "Aux stem fix",
            optimizer_fix_smoke.get("status", "unknown"),
            optimizer_fix_smoke.get("message", ""),
            optimizer_fix_smoke.get("status", "unknown"),
        ),
        (
            "LOD acquisition",
            acquisition.get("lod_status", "unknown"),
            acquisition.get("lod_message", ""),
            acquisition.get("lod_status", "unknown"),
        ),
    ]

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PerceptionISP Achievement Report</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --paper: #ffffff;
      --ink: #17202a;
      --muted: #5f6f82;
      --line: #d9e2ec;
      --blue: #1f6feb;
      --green: #137a45;
      --amber: #9a6700;
      --red: #b42318;
      --steel: #2f3b4a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      background: var(--paper);
      border-bottom: 1px solid var(--line);
      padding: 30px 34px 22px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 31px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 22px 0 10px; font-size: 17px; }}
    p {{ line-height: 1.58; }}
    code {{ background: #eef3f8; border-radius: 4px; padding: 1px 5px; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .subtitle {{ max-width: 1180px; margin: 0; color: var(--muted); line-height: 1.55; }}
    .wrap {{ padding: 18px 34px 42px; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 118px; }}
    .card .label {{ color: var(--muted); font-size: 13px; }}
    .card .value {{ font-size: 24px; font-weight: 740; margin-top: 6px; }}
    .card .detail {{ color: var(--muted); margin-top: 8px; line-height: 1.45; }}
    .status {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid var(--line); background: #eef3f8; }}
    .pass, .supported, .present {{ color: var(--green); border-color: #b7e3c7; background: #eefaf2; }}
    .partial, .pending, .running, .unknown {{ color: var(--amber); border-color: #ead18a; background: #fff8df; }}
    .fail, .missing, .blocked {{ color: var(--red); border-color: #f1b8b3; background: #fff1f0; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; border-bottom: 1px solid var(--line); margin-top: 20px; }}
    .tab-button {{ border: 1px solid var(--line); border-bottom: 0; background: #edf2f7; color: var(--steel); padding: 10px 13px; border-radius: 8px 8px 0 0; cursor: pointer; font-weight: 680; }}
    .tab-button.active {{ background: var(--paper); color: var(--blue); }}
    .tab-panel {{ display: none; background: var(--paper); border: 1px solid var(--line); border-top: 0; border-radius: 0 0 8px 8px; padding: 20px; }}
    .tab-panel.active {{ display: block; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .flow {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 14px 0 6px; }}
    .node {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfe; min-height: 114px; }}
    .node strong {{ display: block; margin-bottom: 7px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f9; }}
    ul {{ line-height: 1.56; }}
    .note {{ border-left: 4px solid var(--blue); background: #f2f7ff; padding: 12px 14px; margin: 14px 0; }}
    .warn {{ border-left-color: var(--amber); background: #fff9e8; }}
    .bad {{ border-left-color: var(--red); background: #fff2f1; }}
    .small {{ color: var(--muted); font-size: 13px; }}
    @media (max-width: 1180px) {{ .cards, .grid2, .flow {{ grid-template-columns: 1fr; }} .wrap, header {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
<header>
  <h1>PerceptionISP Achievement Report</h1>
  <p class="subtitle">이 문서는 지금까지 우리가 만든 PerceptionISP의 목적, 구현 아키텍처, 검증 데이터, 실제 성과, 아직 검증하지 못한 주장과 그 이유를 하나의 standalone HTML로 정리한 자료입니다. 핵심은 “무엇을 자랑할 수 있는가”와 “아직 주장하면 안 되는 경계”를 분리하는 것입니다.</p>
  <p class="small">Generated: {generated} · Overall status: <span class="status {_esc(report.get('status'))}">{_esc(report.get('status'))}</span></p>
</header>
<div class="wrap">
  <section class="cards">{_cards(cards)}</section>
  <nav class="tabs" aria-label="Report tabs">
    {_tab("story", "1. Story", True)}
    {_tab("architecture", "2. Architecture", False)}
    {_tab("blocks", "3. Blocks", False)}
    {_tab("datasets", "4. Datasets", False)}
    {_tab("evidence", "5. Evidence", False)}
    {_tab("hard", "6. Hard Slices", False)}
    {_tab("training", "7. Training", False)}
    {_tab("gaps", "8. Gaps", False)}
    {_tab("roadmap", "9. Roadmap", False)}
    {_tab("sources", "10. Sources", False)}
  </nav>

  <section id="story" class="tab-panel active">
    <h2>Storyline</h2>
    <p>HumanISP는 사람이 보기에 자연스러운 RGB를 만드는 방향입니다. PerceptionISP는 같은 RAW/CFA 입력에서 detector가 쓰기 좋은 edge evidence, artifact/noise reliability, PSF/blur confidence, demosaic-aware auxiliary signals를 같이 만들도록 설계했습니다.</p>
    <div class="note">현재 가장 강하게 말할 수 있는 성과는 네 가지입니다. 첫째, native RAW 프로토콜에서 HumanISP 대비 FP/precision 개선 가능성을 보였습니다. 둘째, RGB+Aux DNN이 aux 채널을 실제로 사용한다는 ablation evidence를 만들었습니다. 셋째, edge/PSF/hard-object 검증 장치를 만들어 어떤 조건에서 PerceptionISP가 의미 있는지 확인하는 구조를 갖췄습니다. 넷째, optimizer/freeze path를 수정한 뒤 gate -1 corrected runs에서 RGB continued control 대비 full-val mAP50-95가 개선됐고, small/thin hard-slice localization 품질도 반복 개선되는 evidence를 확보했습니다.</div>
    <div class="note warn">주의할 점: seed3와 seed4 corrected runs는 수치가 완전히 동일하게 재현됐습니다. 이것은 corrected path의 repeatability 증거로는 좋지만, 독립적인 3-seed 통계로 주장하면 안 됩니다. 진짜 variance 검증은 split/shuffle/augmentation/initialization 중 실제 stochastic source가 달라지는 protocol이 필요합니다.</div>
    <h3>목적에서 성과까지</h3>
    <table>{_rows(_story_rows(), ("Question", "Engineering action", "Current answer"))}</table>
    <h3>현재 claim ladder</h3>
    <table>{_rows(_claim_rows(native, dense, yolo, hard_primary, acquisition), ("Claim", "Status", "Evidence", "Boundary"))}</table>
  </section>

  <section id="architecture" class="tab-panel">
    <h2>Architecture</h2>
    <div class="flow">
      <div class="node"><strong>Scene / Native RAW</strong>CameraE2E scene simulation, PascalRAW native NEF, and future LOD/AODRaw low-light/adverse RAW sources.</div>
      <div class="node"><strong>Sensor Model</strong>CFA mosaic, source CFA provenance, LensPSF blur, noise/exposure stress, and native/remapped-source tracking.</div>
      <div class="node"><strong>HumanISP</strong>Baseline RGB branch for conventional perception pipeline comparison and RGB edge proxy measurement.</div>
      <div class="node"><strong>PerceptionISP</strong>Machine-facing RGB plus aux maps: edge strength, edge evidence, blur/PSF, noise/artifact/reliability signals.</div>
      <div class="node"><strong>Detector / Gate</strong>Fusion calibration, compact RGB+Aux DNN, YOLO RGB+Aux fine-tuning, held-out native/hard-slice reports.</div>
    </div>
    <h3>왜 이런 구조인가</h3>
    <p>Aux map은 object class label을 직접 예측하는 독립 detector가 아닙니다. 그래서 aux map만 detector처럼 평가하면 label-aware mAP가 거의 0에 가깝습니다. 올바른 구조는 RGB detector의 semantic label/proposal을 유지하면서 aux evidence를 calibration 또는 DNN input으로 넣는 것입니다.</p>
    <div class="note warn">잘못 주장하면 안 되는 부분: “aux map 자체가 detector를 대체한다”가 아닙니다. 현재 방향은 “기존 RGB detector가 판단하기 어려운 edge/artifact/blur case에서 aux evidence가 RGB 판단을 보조한다”입니다.</div>
    <h3>Expected effects</h3>
    <table>{_rows(_expected_effect_rows(scene, yolo, hard_primary), ("Component", "Signal", "Expected perception effect", "Current validation"))}</table>
  </section>

  <section id="blocks" class="tab-panel">
    <h2>Implemented Blocks</h2>
    <table>{_rows(_block_rows(), ("Block", "What is implemented", "How it is validated"))}</table>
    <div class="note warn">여기서 “구현됨”은 SW simulation/evaluation pipeline에서 실제로 생성하고 테스트할 수 있다는 뜻입니다. production ISP quality, all-sensor robustness, all-detector superiority를 의미하지는 않습니다.</div>
  </section>

  <section id="datasets" class="tab-panel">
    <h2>Datasets and Test Protocol</h2>
    <table>{_rows(_dataset_rows(native, dense, scene, yolo, hard_primary, acquisition), ("Dataset / protocol", "Purpose", "What we tested", "Current limitation"))}</table>
    <h3>Dataset acquisition state</h3>
    <table>{_rows(_acquisition_rows(acquisition), ("Item", "Status", "Meaning"))}</table>
  </section>

  <section id="evidence" class="tab-panel">
    <h2>Evidence Summary</h2>
    <div class="grid2">
      <div>
        <h3>Native RAW held-out gate</h3>
        <table>{_rows(_native_rows(native))}</table>
      </div>
      <div>
        <h3>RGB+Aux input ablation</h3>
        <table>{_rows(_dense_rows(dense))}</table>
      </div>
    </div>
    <h3>Scene-edge / PSF diagnostics</h3>
    <table>{_rows(_scene_rows(scene))}</table>
    <h3>Matched YOLO seed2 full-val result: plain edge6</h3>
    <table>{_rows(_yolo_rows(yolo))}</table>
    <h3>Matched YOLO seed2 full-val result: gated edge6</h3>
    <table>{_rows(_yolo_rows(gated))}</table>
    <div class="note warn">Matched YOLO seed2는 recall이 좋아졌지만 mAP50-95가 낮아졌습니다. 따라서 이 결과만으로 “PerceptionISP가 RGB-only보다 우월하다”고 말하면 안 됩니다. 의미 있는 해석은 “aux path가 recall signal을 만들었고, hard-case targeting 또는 better pretraining이 필요하다”입니다.</div>
  </section>

  <section id="hard" class="tab-panel">
    <h2>Hard-Object Slice Results</h2>
    <p>PerceptionISP의 edge aux가 정말 필요하다면 전체 val보다 small object, thin/long object, weak edge, blur/low-MTF 같은 hard slice에서 먼저 신호가 보여야 합니다.</p>
    <h3>Matched hardx3 fine-tune: gated edge6 vs RGB-only</h3>
    <table>{_rows(_hard_rows(hard_hardx3), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note warn">Hard-case oversampling은 RGB-only 쪽 small-object 성능을 개선했지만, 같은 recipe의 gated Aux fine-tune은 small slice를 크게 잃고 thin slice에서만 AP 개선을 보였습니다. 따라서 현재 결론은 “PerceptionISP가 전체 PascalRAW에서 우월하다”가 아니라 “thin/edge-like object에는 신호가 있으나 aux stem 안정화와 small-object 특화 학습이 필요하다”입니다.</div>
    <h3>RGB-preserved Aux stem: hardx3 RGB checkpoint to gated Aux</h3>
    <table>{_rows(_hard_rows(hard_freezergb), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note">이 variant는 RGB branch를 보존/freeze하고 aux branch만 gated로 작게 더해, RGB baseline을 무너뜨리지 않는지 확인하기 위한 안정화 실험입니다. 단, RGB hardx3 checkpoint에서 추가 epoch를 돌리므로 최종 claim에는 RGB-only continued-training control도 필요합니다.</div>
    <h3>Extra-training control: RGB-preserved Aux vs RGB-only continued</h3>
    <table>{_rows(_hard_rows(hard_freezergb_vs_continue), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note warn">이 표가 가장 공정한 다음 판정입니다. RGB-only도 같은 checkpoint에서 같은 추가 학습을 받았기 때문에, 여기서 이기지 못하면 개선은 Aux 때문이라고 주장하기 어렵습니다.</div>
    <h3>Corrected fixedopt gate3: Aux vs RGB-only continued</h3>
    <table>{_rows(_hard_rows(hard_fixedopt_gate3), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note">이 run은 aux stem이 optimizer 생성 전에 설치되고, RGB branch freeze가 재적용되는 수정 후 결과입니다. full-val broad win은 아니지만 hard slice에서 mAP50-95와 fixed-threshold precision이 소폭 개선됐습니다. 단, gate sigmoid가 거의 0.047로 유지되어 aux 활용은 아직 너무 약합니다.</div>
    <h3>Corrected fixedopt gate1 screening: Aux vs RGB-only continued</h3>
    <table>{_rows(_hard_rows(hard_fixedopt_gate1), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note">gate -1 screening은 gate -3보다 aux contribution을 더 열어둔 corrected run입니다. full-val mAP50-95가 RGB continued control보다 좋아졌고, hard slice에서는 mAP50-95와 fixed-threshold precision이 개선됐습니다. 다만 2 epoch / seed2 screening이므로 final claim이 아니라 다음 4 epoch multi-seed run으로 확장할 후보입니다.</div>
    <h3>Corrected fixedopt gate1 e4: Aux vs RGB-only continued</h3>
    <table>{_rows(_hard_rows(hard_fixedopt_gate1_e4), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note warn">gate -1 e4의 best.pt는 2 epoch screening과 같은 최종 metric을 냈고, 3 epoch validation에서는 mAP50-95가 떨어졌습니다. 따라서 좋은 신호는 “gate -1 corrected path가 유효하다”이지 “더 오래 학습하면 계속 좋아진다”가 아닙니다. 다음 run은 2 epoch 또는 early stopping/short LR schedule로 잡는 편이 맞습니다.</div>
    <h3>Corrected fixedopt gate1 e2 seed3: Aux vs RGB-only continued</h3>
    <table>{_rows(_hard_rows(hard_fixedopt_gate1_e2_seed3), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note">seed3 반복에서는 full-val mAP50-95가 RGB control 대비 +0.0068 개선됐고, small/thin hard slice mAP50-95도 양수였습니다. 이 결과는 seed2에서 본 precision/localization 개선이 단일 seed 우연만은 아닐 가능성을 높입니다. 다만 recall tradeoff가 남아 있어 recall superiority claim으로 쓰면 안 됩니다.</div>
    <h3>Corrected fixedopt gate1 e2 seed4 repeat: Aux vs RGB-only continued</h3>
    <table>{_rows(_hard_rows(hard_fixedopt_gate1_e2_seed4), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note warn">seed4는 seed3와 동일한 full-val/hard-slice result를 재현했습니다. 따라서 evidence 해석은 “repeatable corrected path”가 맞고, “statistically independent multi-seed win”은 아직 아닙니다.</div>
    <h3>Hardx5 oversampling stress: Aux freeze-RGB vs RGB control</h3>
    <table>{_rows(_hard_rows(hard_hardx5), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <div class="note bad">Hardx5는 중요한 반례입니다. full-val precision/recall은 약간 좋아졌지만 hard-object AP와 localization 품질은 악화됐습니다. 따라서 단순 oversampling 강화만으로 PerceptionISP 장점을 주장하면 안 됩니다.</div>
    <h3>Fair seed2: gated edge6 vs RGB-only</h3>
    <table>{_rows(_hard_rows(hard_gated), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <h3>Fair seed2: plain edge6 vs RGB-only</h3>
    <table>{_rows(_hard_rows(hard), ("Slice", "Target GT", "dAP50", "dAP50-95", "dP@0.25", "dR@0.25", "Interpretation"))}</table>
    <h3>Gated / hard-mining candidates</h3>
    <table>{_rows(_candidate_rows(candidates), ("Pair", "Recipe fairness", "All dAP50-95", "Small dAP50-95", "Thin dAP50-95", "Small|thin dAP50-95", "Use as claim?"))}</table>
    <div class="note warn">Seed1 gated512 결과는 수치상 매우 유망하지만, RGB baseline과 batch/augmentation/lrf recipe가 달라 fair claim으로 쓰면 안 됩니다. 이것은 “다시 matched recipe로 검증해야 하는 후보”입니다.</div>
  </section>

  <section id="training" class="tab-panel">
    <h2>Training Status</h2>
    <h3>Completed matched runs</h3>
    <table>{_rows(_training_rows(yolo, gated, hardx3_train, hardx3_freezergb_train, hardx3_continue_train, hardx3_freezergb_vs_continue_train, hardx5_train, fixedopt_gate3_train, fixedopt_gate1_train, fixedopt_gate1_e4_train, fixedopt_gate1_e2_seed3_train, fixedopt_gate1_e2_seed4_train), ("Run", "Status", "Precision", "Recall", "mAP50", "mAP50-95", "Delta vs RGB"))}</table>
    <h3>Corrected aux gate state</h3>
    <table>{_rows(_aux_gate_rows(fixedopt_gate3_train, fixedopt_gate1_train, fixedopt_gate1_e4_train, fixedopt_gate1_e2_seed3_train, fixedopt_gate1_e2_seed4_train), ("Run", "Gate sigmoid mean", "RGB branch frozen", "Interpretation"))}</table>
    <h3>Latest training-path correction</h3>
    <table>{_rows(_optimizer_fix_rows(optimizer_fix_smoke))}</table>
    <div class="note bad">최근 발견한 중요한 문제: 이전 Aux stem 일부 실험은 gate/aux branch가 optimizer에 늦게 붙어 실제 학습되지 않았을 가능성이 있습니다. 그래서 그 결과를 “learned aux branch 성능”으로 강하게 주장하면 안 됩니다. 현재 wrapper는 stem을 optimizer 생성 전에 설치하고, freeze 보장도 callback으로 다시 확인하도록 수정했습니다.</div>
    <h3>Why we are training RGB+Aux</h3>
    <ul>
      <li>기존 RGB DNN을 그대로 쓰면 aux map이 network input에 들어가지 않으므로 DNN이 aux를 학습적으로 사용할 수 없습니다.</li>
      <li>현재 구조는 RGB input stem을 RGB+Aux stem으로 확장하고, 가능하면 기존 RGB feature behavior를 유지하면서 aux branch만 추가 학습하는 방향입니다.</li>
      <li>Aux feature pretraining은 aux-only feature가 RGB first feature와 비슷한 표현을 만들도록 초기화하는 목적입니다. 작은 데이터에서 random aux branch보다 빠르고 안정적인 시작점이 됩니다.</li>
    </ul>
    <div class="note">Hardx3 matched fine-tune까지 완료됐고, 결과는 RGB-only hardx3가 더 강했습니다. 다음 실험은 freeze/gate/aux pretraining variant 또는 small/thin focused sampling으로 좁히는 것이 맞습니다.</div>
  </section>

  <section id="gaps" class="tab-panel">
    <h2>Not Yet Proven</h2>
    <table>{_rows(_gap_rows(), ("Unverified claim", "Why not proven yet", "How to verify", "Expected likelihood"))}</table>
    <h3>검증하지 못한 이유에 대한 고찰</h3>
    <p>PascalRAW는 지금 가장 현실적으로 쓸 수 있는 native RAW detection dataset이지만, 일반 장면 비중이 높아 small/thin/low-light/blur 같은 PerceptionISP 유리 조건이 희석됩니다. 반대로 AODRaw/LOD는 더 의미 있는 adverse RAW 후보지만 접근성, 용량, 다운로드 안정성, 실제 native RAW 여부 확인이 필요합니다. 그러므로 지금은 PascalRAW로 pipeline correctness와 fair comparison을 고정하고, LOD로 low-light evidence를 추가하는 순서가 맞습니다.</p>
  </section>

  <section id="roadmap" class="tab-panel">
    <h2>Roadmap</h2>
    <table>{_rows(_roadmap_rows(), ("Priority", "Action", "Output evidence", "Decision gate"))}</table>
  </section>

  <section id="sources" class="tab-panel">
    <h2>Sources</h2>
    <p>이 보고서는 로컬 실험 산출물과 summary JSON에서 생성되었습니다.</p>
    <h3>Report summaries</h3>
    <table>{_rows(_source_rows(_map(report.get('summaries'))), ("Key", "Summary JSON", "Expected report"))}</table>
    <h3>Training summaries</h3>
    <table>{_rows(_source_rows(_map(report.get('train_summaries'))), ("Key", "Summary JSON", "Expected report"))}</table>
  </section>
</div>
<script>
  const buttons = Array.from(document.querySelectorAll('.tab-button'));
  const panels = Array.from(document.querySelectorAll('.tab-panel'));
  function activate(id) {{
    buttons.forEach(button => button.classList.toggle('active', button.dataset.tab === id));
    panels.forEach(panel => panel.classList.toggle('active', panel.id === id));
    history.replaceState(null, '', '#' + id);
  }}
  buttons.forEach(button => button.addEventListener('click', () => activate(button.dataset.tab)));
  const initial = location.hash ? location.hash.slice(1) : 'story';
  if (document.getElementById(initial)) activate(initial);
</script>
</body>
</html>
"""


def _story_rows() -> list[tuple[str, str, str]]:
    return [
        (
            "왜 HumanISP가 아니라 PerceptionISP인가?",
            "사람이 보기 좋은 RGB와 detector가 쓰기 좋은 신호를 분리해 비교 가능한 pipeline을 구축했습니다.",
            "Native RAW에서 FP/precision 개선 가능성을 보였고, aux evidence를 detector input/fusion으로 연결했습니다.",
        ),
        (
            "Aux map이 실제로 쓸모가 있는가?",
            "RGB+Aux tensor export, compact DNN training, input ablation, YOLO aux stem training을 구현했습니다.",
            "Compact DNN은 aux를 실제로 사용합니다. YOLO full-val에서는 recall gain은 있으나 AP 품질 개선은 아직 부족합니다.",
        ),
        (
            "어떤 조건에서 효과가 클 것으로 보는가?",
            "small/thin object, low-MTF blur, demosaic artifact, weak edge, low light, CFA/PSF-sensitive slice를 분리했습니다.",
            "Scene-edge/PSF diagnostic은 방향성이 맞고, hard-object slice 평가가 시작됐습니다.",
        ),
        (
            "다음 성능개선의 핵심은 무엇인가?",
            "Fair matched training, hard-slice mining, LOD low-light RAW validation, better aux pretraining 순서로 좁힙니다.",
            "큰 리포트보다 개선 loop가 우선입니다. 보고서는 이 loop의 증거를 자동 정리하는 역할입니다.",
        ),
    ]


def _claim_rows(
    native: Mapping[str, Any],
    dense: Mapping[str, Any],
    yolo: Mapping[str, Any],
    hard: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> list[tuple[str, str, str, str]]:
    return [
        (
            "Software feasibility",
            "Supported",
            "RAW/CFA ingestion, HumanISP, PerceptionISP, aux export, detector gates, YOLO RGB+Aux path, reports are implemented.",
            "Production ISP quality is separate from software feasibility.",
        ),
        (
            "Native RAW FP/precision benefit",
            "Supported in PascalRAW RGGB-heavy gate",
            f"dP50={_fmt(native.get('precision50_delta'), signed=True)}, dFP50={_fmt(native.get('fp50_delta'), signed=True)}, dR50={_fmt(native.get('recall50_delta'), signed=True)}.",
            "Not all CFA/adverse-condition proof.",
        ),
        (
            "DNN uses aux input",
            "Supported for compact DNN",
            f"zero-aux recall delta={_fmt(dense.get('zero_aux_recall_delta'), signed=True)} across {_fmt(dense.get('seed_count'), digits=0)} seeds.",
            "Not yet enough for production detector superiority.",
        ),
        (
            "YOLO RGB+Aux full-val improvement",
            "Partially tested; not broadly supported",
            f"Matched seed2 dR={_fmt(yolo.get('recall_delta'), signed=True)}, dAP50-95={_fmt(yolo.get('map5095_delta'), signed=True)}.",
            "Recall gain exists, but localization/AP quality is weaker.",
        ),
        (
            "Hard-slice edge advantage",
            "Open",
            _hard_claim_text(hard),
            "Thin-object AP quality shows a weak positive signal, but small-object performance is worse. Need stronger hard mining and low-light/adverse data.",
        ),
        (
            "Low-light/adverse RAW advantage",
            "Pending",
            f"LOD={acquisition.get('lod_status', 'unknown')}: {acquisition.get('lod_message', '')}",
            "Need complete dataset and native RAW inspection.",
        ),
    ]


def _expected_effect_rows(
    scene: Mapping[str, Any], yolo: Mapping[str, Any], hard: Mapping[str, Any]
) -> list[tuple[str, str, str, str]]:
    return [
        (
            "CFA provenance",
            "source CFA, target CFA, native/remapped flag, source resolution",
            "Prevents invalid comparisons caused by silent CFA mismatch or RGB-proxy RAW.",
            "Used in native PascalRAW gate; all-CFA claim still pending.",
        ),
        (
            "Perception RGB",
            "edge-preserving demosaic/tone path",
            "Can reduce false edges/artifacts while keeping object boundaries useful.",
            f"Scene edge dF1 vs Human={_fmt(scene.get('perception_rgb_d_f1'), signed=True)}.",
        ),
        (
            "Aux edge evidence",
            "edge strength/evidence channels exposed to DNN or fusion gate",
            "Should help weak boundaries, small/thin objects, blur and demosaic ambiguity.",
            f"Matched YOLO seed2 recall delta={_fmt(yolo.get('recall_delta'), signed=True)}; hard slice still not proven.",
        ),
        (
            "Noise/artifact reliability",
            "confidence channels for noisy or artifact-prone regions",
            "Can suppress proposals caused by sensor or ISP artifacts.",
            "Native gate FP reduction supports this direction, but condition-specific proof is pending.",
        ),
        (
            "LensPSF-aware evidence",
            "blur/PSF-sensitive edge confidence",
            "Can expose optical ambiguity when PSF is meaningful.",
            f"Aux/scene diagnostic best dF1={_fmt(scene.get('aux_best_d_f1'), signed=True)}; stronger PSF cases needed.",
        ),
        (
            "Hard-slice gate",
            "small/thin object metrics separated from easy full-val",
            "Prevents easy objects from hiding the value of edge aux.",
            _hard_claim_text(hard),
        ),
    ]


def _block_rows() -> list[tuple[str, str, str]]:
    return [
        ("RAW/CFA ingestion", "PascalRAW NEF, CameraE2E bridge, CFA provenance, native source checks.", "Native audit and CFA provenance summaries."),
        ("HumanISP baseline", "RGB-facing demosaic/tone baseline used as detector and edge comparison.", "HumanISP vs PerceptionISP gates."),
        ("PerceptionISP RGB", "Edge-aware demosaic/denoise/tone/artifact-aware RGB branch.", "Scene-edge and native detector gates."),
        ("Aux maps", "edge strength, edge evidence, PSF/blur, noise/artifact/reliability style maps.", "Aux sweep, edge6 export, DNN ablations."),
        ("RGB+Aux export", "NPZ/YOLO dataset export with explicit channel names and edge6 validation.", "YOLO edge6 audit and matched training summaries."),
        ("Fusion/calibration", "RGB detector class label retained while aux evidence calibrates proposal confidence.", "Native claim dashboard and FP/precision gate."),
        ("DNN training", "Compact RGB+Aux training, YOLO aux stem, gated_sum aux stem, adapter initialization.", "Train summaries and ablation reports."),
        ("Dataset acquisition", "PascalRAW in use, LOD/AODRaw local-state/readiness tracking.", "Raw acquisition and LOD readiness reports."),
    ]


def _dataset_rows(
    native: Mapping[str, Any],
    dense: Mapping[str, Any],
    scene: Mapping[str, Any],
    yolo: Mapping[str, Any],
    hard: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> list[tuple[str, str, str, str]]:
    return [
        (
            "PascalRAW native held-out 750",
            "Main HumanISP vs PerceptionISP native RAW claim gate.",
            f"dP50={_fmt(native.get('precision50_delta'), signed=True)}, dFP50={_fmt(native.get('fp50_delta'), signed=True)}, dR50={_fmt(native.get('recall50_delta'), signed=True)}.",
            "Not enough adverse low-light/weather diversity.",
        ),
        (
            "PascalRAW native1581 compact DNN",
            "Prove RGB+Aux model uses aux channels.",
            f"zero-aux recall delta={_fmt(dense.get('zero_aux_recall_delta'), signed=True)}; seeds={_fmt(dense.get('seed_count'), digits=0)}.",
            "Compact DNN, not production detector proof.",
        ),
        (
            "Scene-edge CFA/PSF sweep",
            "Check whether edge information follows richer source scene edge proxy.",
            f"Perception RGB dF1={_fmt(scene.get('perception_rgb_d_f1'), signed=True)}, aux strength dF1={_fmt(scene.get('aux_strength_d_f1'), signed=True)}.",
            "Source-edge proxy, not object boundary GT.",
        ),
        (
            "YOLO seed2 full PascalRAW",
            "Fair RGB-only vs RGB+Aux edge6 detector comparison.",
            f"dR={_fmt(yolo.get('recall_delta'), signed=True)}, dAP50-95={_fmt(yolo.get('map5095_delta'), signed=True)}.",
            "Recall signal but not AP superiority.",
        ),
        (
            "Hard-object slice",
            "Expose small/thin object behavior instead of averaging over easy objects.",
            _hard_claim_text(hard),
            "Gated edge6 gives a slight thin/small|thin AP50-95 signal, but small-object AP is still worse.",
        ),
        (
            "LOD low-light RAW",
            "Next practical low-light dataset for adverse-condition claim.",
            f"{acquisition.get('lod_status', 'unknown')}: {acquisition.get('lod_message', '')}",
            "Download incomplete and archive/native format must be inspected.",
        ),
        (
            "AODRaw",
            "Large adverse object detection dataset candidate.",
            acquisition.get("aodraw_state", "unknown"),
            "Large storage/network cost; local RAW image package not ready.",
        ),
    ]


def _acquisition_rows(acquisition: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    return [
        ("LOD archive", acquisition.get("lod_status", "unknown"), acquisition.get("lod_message", "")),
        ("LOD readiness", acquisition.get("lod_readiness", "unknown"), "Ready only after full archive and image/annotation format check."),
        ("AODRaw", acquisition.get("aodraw_state", "unknown"), "Stronger adverse benchmark, but heavier and currently not locally ready."),
        ("Practical decision", "Use LOD first", "Given disk pressure, LOD is the realistic next low-light RAW validation path."),
    ]


def _native_rows(native: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Samples", _fmt(native.get("sample_count"), digits=0)),
        ("Status", native.get("status", "unknown")),
        ("Claim status", native.get("claim_status", "unknown")),
        ("Precision@0.50 delta", _fmt(native.get("precision50_delta"), signed=True)),
        ("FP@0.50 delta", _fmt(native.get("fp50_delta"), signed=True)),
        ("Recall@0.50 delta", _fmt(native.get("recall50_delta"), signed=True)),
        ("Native RAW accepted fraction", _fmt(native.get("native_raw_fraction"))),
        ("True CFA mosaic fraction", _fmt(native.get("true_cfa_fraction"))),
        ("Source CFA patterns", _json(native.get("source_cfa_patterns"))),
        ("Boundary", native.get("boundary", "")),
    ]


def _dense_rows(dense: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Status", dense.get("status", "unknown")),
        ("Seeds", _fmt(dense.get("seed_count"), digits=0)),
        ("Test samples", _fmt(dense.get("test_sample_count"), digits=0)),
        ("Full RGB+Aux precision", _fmt(dense.get("full_precision"))),
        ("Full RGB+Aux recall", _fmt(dense.get("full_recall"))),
        ("Zero aux recall delta", _fmt(dense.get("zero_aux_recall_delta"), signed=True)),
        ("Zero RGB recall delta", _fmt(dense.get("zero_rgb_recall_delta"), signed=True)),
        ("Shuffle aux recall delta", _fmt(dense.get("shuffle_aux_recall_delta"), signed=True)),
        ("Boundary", dense.get("boundary", "")),
    ]


def _scene_rows(scene: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Scene edge proxy", "High-information source scene edge used as proxy, not object-boundary GT."),
        ("Perception RGB vs Human RGB", f"dF1={_fmt(scene.get('perception_rgb_d_f1'), signed=True)}, win-rate={_fmt(scene.get('perception_rgb_win_rate'))}."),
        ("Aux strength vs Human RGB", f"dF1={_fmt(scene.get('aux_strength_d_f1'), signed=True)}, win-rate={_fmt(scene.get('aux_strength_win_rate'))}."),
        ("Old aux confidence", f"dF1={_fmt(scene.get('aux_confidence_d_f1'), signed=True)}; this is weaker than edge strength/evidence."),
        ("Best aux evidence candidate", f"name={scene.get('aux_best_name', 'unknown')}, dF1={_fmt(scene.get('aux_best_d_f1'), signed=True)}."),
        ("PSF interpretation", "If PSF kernel is too small, effect is expected to be weak; larger meaningful PSF should expose edge ambiguity more clearly."),
    ]


def _yolo_rows(yolo: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Status", yolo.get("status", "unknown")),
        ("Protocol", yolo.get("comparison", "")),
        ("RGB-only", _metric_text(yolo.get("base"))),
        ("RGB+Aux edge6", _metric_text(yolo.get("candidate"))),
        ("Delta", f"dP={_fmt(yolo.get('precision_delta'), signed=True)}, dR={_fmt(yolo.get('recall_delta'), signed=True)}, dAP50={_fmt(yolo.get('map50_delta'), signed=True)}, dAP50-95={_fmt(yolo.get('map5095_delta'), signed=True)}."),
        ("Interpretation", yolo.get("interpretation", "")),
    ]


def _hard_rows(hard: Mapping[str, Any]) -> list[tuple[str, str, str, str, str, str, str]]:
    rows = []
    filters = _map(hard.get("filters"))
    for name in ("all", "small", "thin", "small_or_thin"):
        row = _map(filters.get(name))
        rows.append(
            (
                name,
                _fmt(row.get("target_gt_count"), digits=0),
                _fmt(row.get("map50_delta"), signed=True),
                _fmt(row.get("map5095_delta"), signed=True),
                _fmt(row.get("precision025_delta"), signed=True),
                _fmt(row.get("recall025_delta"), signed=True),
                _hard_interpretation(row),
            )
        )
    if not rows:
        rows.append(("missing", "n/a", "n/a", "n/a", "n/a", "n/a", "Hard eval summary is missing."))
    return rows


def _candidate_rows(candidates: Sequence[Any]) -> list[tuple[str, str, str, str, str, str, str]]:
    rows = []
    for item in candidates:
        row = _map(item)
        filters = _map(row.get("filters"))
        rows.append(
            (
                row.get("label", ""),
                row.get("fairness", ""),
                _fmt(_map(filters.get("all")).get("map5095_delta"), signed=True),
                _fmt(_map(filters.get("small")).get("map5095_delta"), signed=True),
                _fmt(_map(filters.get("thin")).get("map5095_delta"), signed=True),
                _fmt(_map(filters.get("small_or_thin")).get("map5095_delta"), signed=True),
                row.get("claim_use", ""),
            )
        )
    return rows or [("missing", "n/a", "n/a", "n/a", "n/a", "n/a", "No candidate hard eval summary.")]


def _training_rows(
    yolo: Mapping[str, Any],
    gated: Mapping[str, Any],
    hardx3: Mapping[str, Any] | None = None,
    hardx3_freezergb: Mapping[str, Any] | None = None,
    hardx3_continue: Mapping[str, Any] | None = None,
    hardx3_freezergb_vs_continue: Mapping[str, Any] | None = None,
    hardx5: Mapping[str, Any] | None = None,
    fixedopt_gate3: Mapping[str, Any] | None = None,
    fixedopt_gate1: Mapping[str, Any] | None = None,
    fixedopt_gate1_e4: Mapping[str, Any] | None = None,
    fixedopt_gate1_e2_seed3: Mapping[str, Any] | None = None,
    fixedopt_gate1_e2_seed4: Mapping[str, Any] | None = None,
) -> list[tuple[str, str, str, str, str, str, str]]:
    base = _map(yolo.get("base"))
    edge = _map(yolo.get("candidate"))
    gated_candidate = _map(gated.get("candidate"))
    rows = [
        ("RGB-only seed2", base.get("status", "complete"), _fmt(base.get("precision")), _fmt(base.get("recall")), _fmt(base.get("map50")), _fmt(base.get("map5095")), "baseline"),
        (
            "Plain RGB+Aux edge6 seed2",
            edge.get("status", "complete"),
            _fmt(edge.get("precision")),
            _fmt(edge.get("recall")),
            _fmt(edge.get("map50")),
            _fmt(edge.get("map5095")),
            f"dR={_fmt(yolo.get('recall_delta'), signed=True)}, dAP50-95={_fmt(yolo.get('map5095_delta'), signed=True)}",
        ),
        (
            "Gated RGB+Aux edge6 seed2 matched",
            gated_candidate.get("status", gated.get("status", "unknown")),
            _fmt(gated_candidate.get("precision")),
            _fmt(gated_candidate.get("recall")),
            _fmt(gated_candidate.get("map50")),
            _fmt(gated_candidate.get("map5095")),
            gated.get("delta_text", gated.get("message", "final summary pending")),
        ),
    ]
    if hardx3 is not None:
        hardx3_candidate = _map(hardx3.get("candidate"))
        rows.append(
            (
                "Gated RGB+Aux hardx3 seed2 matched",
                hardx3_candidate.get("status", hardx3.get("status", "unknown")),
                _fmt(hardx3_candidate.get("precision")),
                _fmt(hardx3_candidate.get("recall")),
                _fmt(hardx3_candidate.get("map50")),
                _fmt(hardx3_candidate.get("map5095")),
                hardx3.get("delta_text", hardx3.get("message", "final summary pending")),
            )
        )
    if hardx3_freezergb is not None:
        hardx3_freezergb_candidate = _map(hardx3_freezergb.get("candidate"))
        rows.append(
            (
                "RGB hardx3 -> gated Aux freeze-RGB seed2",
                hardx3_freezergb_candidate.get("status", hardx3_freezergb.get("status", "unknown")),
                _fmt(hardx3_freezergb_candidate.get("precision")),
                _fmt(hardx3_freezergb_candidate.get("recall")),
                _fmt(hardx3_freezergb_candidate.get("map50")),
                _fmt(hardx3_freezergb_candidate.get("map5095")),
                hardx3_freezergb.get("delta_text", hardx3_freezergb.get("message", "final summary pending")),
            )
        )
    if hardx3_continue is not None:
        hardx3_continue_candidate = _map(hardx3_continue.get("candidate"))
        rows.append(
            (
                "RGB hardx3 -> RGB continued seed2 control",
                hardx3_continue_candidate.get("status", hardx3_continue.get("status", "unknown")),
                _fmt(hardx3_continue_candidate.get("precision")),
                _fmt(hardx3_continue_candidate.get("recall")),
                _fmt(hardx3_continue_candidate.get("map50")),
                _fmt(hardx3_continue_candidate.get("map5095")),
                hardx3_continue.get("delta_text", hardx3_continue.get("message", "final summary pending")),
            )
        )
    if hardx3_freezergb_vs_continue is not None:
        hardx3_freezergb_vs_continue_candidate = _map(hardx3_freezergb_vs_continue.get("candidate"))
        rows.append(
            (
                "Gated Aux freeze-RGB vs RGB continued control",
                hardx3_freezergb_vs_continue_candidate.get(
                    "status", hardx3_freezergb_vs_continue.get("status", "unknown")
                ),
                _fmt(hardx3_freezergb_vs_continue_candidate.get("precision")),
                _fmt(hardx3_freezergb_vs_continue_candidate.get("recall")),
                _fmt(hardx3_freezergb_vs_continue_candidate.get("map50")),
                _fmt(hardx3_freezergb_vs_continue_candidate.get("map5095")),
                hardx3_freezergb_vs_continue.get(
                    "delta_text", hardx3_freezergb_vs_continue.get("message", "final summary pending")
                ),
            )
        )
    if hardx5 is not None:
        hardx5_candidate = _map(hardx5.get("candidate"))
        rows.append(
            (
                "Hardx5 Aux freeze-RGB vs RGB control",
                hardx5_candidate.get("status", hardx5.get("status", "unknown")),
                _fmt(hardx5_candidate.get("precision")),
                _fmt(hardx5_candidate.get("recall")),
                _fmt(hardx5_candidate.get("map50")),
                _fmt(hardx5_candidate.get("map5095")),
                hardx5.get("delta_text", hardx5.get("message", "final summary pending")),
            )
        )
    if fixedopt_gate3 is not None:
        fixedopt_candidate = _map(fixedopt_gate3.get("candidate"))
        rows.append(
            (
                "Corrected gated Aux fixedopt gate3 vs RGB continued",
                fixedopt_candidate.get("status", fixedopt_gate3.get("status", "unknown")),
                _fmt(fixedopt_candidate.get("precision")),
                _fmt(fixedopt_candidate.get("recall")),
                _fmt(fixedopt_candidate.get("map50")),
                _fmt(fixedopt_candidate.get("map5095")),
                fixedopt_gate3.get("delta_text", fixedopt_gate3.get("message", "final summary pending")),
            )
        )
    if fixedopt_gate1 is not None:
        fixedopt_candidate = _map(fixedopt_gate1.get("candidate"))
        rows.append(
            (
                "Corrected gated Aux fixedopt gate1 screen vs RGB continued",
                fixedopt_candidate.get("status", fixedopt_gate1.get("status", "unknown")),
                _fmt(fixedopt_candidate.get("precision")),
                _fmt(fixedopt_candidate.get("recall")),
                _fmt(fixedopt_candidate.get("map50")),
                _fmt(fixedopt_candidate.get("map5095")),
                fixedopt_gate1.get("delta_text", fixedopt_gate1.get("message", "final summary pending")),
            )
        )
    if fixedopt_gate1_e4 is not None:
        fixedopt_candidate = _map(fixedopt_gate1_e4.get("candidate"))
        rows.append(
            (
                "Corrected gated Aux fixedopt gate1 e4 vs RGB continued",
                fixedopt_candidate.get("status", fixedopt_gate1_e4.get("status", "unknown")),
                _fmt(fixedopt_candidate.get("precision")),
                _fmt(fixedopt_candidate.get("recall")),
                _fmt(fixedopt_candidate.get("map50")),
                _fmt(fixedopt_candidate.get("map5095")),
                fixedopt_gate1_e4.get("delta_text", fixedopt_gate1_e4.get("message", "final summary pending")),
            )
        )
    if fixedopt_gate1_e2_seed3 is not None:
        fixedopt_candidate = _map(fixedopt_gate1_e2_seed3.get("candidate"))
        rows.append(
            (
                "Corrected gated Aux fixedopt gate1 e2 seed3 vs RGB continued",
                fixedopt_candidate.get("status", fixedopt_gate1_e2_seed3.get("status", "unknown")),
                _fmt(fixedopt_candidate.get("precision")),
                _fmt(fixedopt_candidate.get("recall")),
                _fmt(fixedopt_candidate.get("map50")),
                _fmt(fixedopt_candidate.get("map5095")),
                fixedopt_gate1_e2_seed3.get(
                    "delta_text", fixedopt_gate1_e2_seed3.get("message", "final summary pending")
                ),
            )
        )
    if fixedopt_gate1_e2_seed4 is not None:
        fixedopt_candidate = _map(fixedopt_gate1_e2_seed4.get("candidate"))
        rows.append(
            (
                "Corrected gated Aux fixedopt gate1 e2 seed4 repeat vs RGB continued",
                fixedopt_candidate.get("status", fixedopt_gate1_e2_seed4.get("status", "unknown")),
                _fmt(fixedopt_candidate.get("precision")),
                _fmt(fixedopt_candidate.get("recall")),
                _fmt(fixedopt_candidate.get("map50")),
                _fmt(fixedopt_candidate.get("map5095")),
                fixedopt_gate1_e2_seed4.get(
                    "delta_text", fixedopt_gate1_e2_seed4.get("message", "final summary pending")
                ),
            )
        )
    return rows


def _aux_gate_rows(
    fixedopt_gate3: Mapping[str, Any],
    fixedopt_gate1: Mapping[str, Any],
    fixedopt_gate1_e4: Mapping[str, Any],
    fixedopt_gate1_e2_seed3: Mapping[str, Any] | None = None,
    fixedopt_gate1_e2_seed4: Mapping[str, Any] | None = None,
) -> list[tuple[str, str, str, str]]:
    rows = [
        _aux_gate_row("fixedopt gate3", fixedopt_gate3),
        _aux_gate_row("fixedopt gate1 screen", fixedopt_gate1),
        _aux_gate_row("fixedopt gate1 e4", fixedopt_gate1_e4),
    ]
    if fixedopt_gate1_e2_seed3 is not None:
        rows.append(_aux_gate_row("fixedopt gate1 e2 seed3", fixedopt_gate1_e2_seed3))
    if fixedopt_gate1_e2_seed4 is not None:
        rows.append(_aux_gate_row("fixedopt gate1 e2 seed4 repeat", fixedopt_gate1_e2_seed4))
    return rows


def _aux_gate_row(label: str, comparison: Mapping[str, Any]) -> tuple[str, str, str, str]:
    state = _map(comparison.get("candidate_aux_stem_state_after_train"))
    stem = _map(comparison.get("candidate_aux_stem"))
    freeze = _map(stem.get("freeze_enforced_after_ultralytics_freeze"))
    gate = state.get("gate_sigmoid_mean")
    rgb_frozen = (
        state.get("rgb_branch_requires_grad")
        if state.get("rgb_branch_requires_grad") is not None
        else freeze.get("requires_grad_after")
    )
    gate_value = _float(gate)
    if state.get("status") != "present":
        interpretation = "No final gate state recorded."
    elif gate_value < 0.1:
        interpretation = "Aux path stayed almost closed; use mainly as corrected-path smoke, not strong aux-use evidence."
    elif gate_value < 0.4:
        interpretation = "Aux path is open enough for screening; promote only after longer/multi-seed confirmation."
    else:
        interpretation = "Aux path is strongly open; check overfit and RGB degradation before claim use."
    return (label, _fmt(gate), _json(rgb_frozen), interpretation)


def _optimizer_fix_rows(smoke: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Status", smoke.get("status", "unknown")),
        ("Message", smoke.get("message", "")),
        ("Aux stem setup phase", smoke.get("setup_phase", "unknown")),
        ("Freeze enforcement", smoke.get("freeze_status", "unknown")),
        ("Final gate sigmoid mean", _fmt(smoke.get("gate_sigmoid_mean"))),
        ("RGB branch requires grad", _json(smoke.get("rgb_branch_requires_grad"))),
        ("Interpretation", smoke.get("interpretation", "")),
    ]


def _gap_rows() -> list[tuple[str, str, str, str]]:
    return [
        (
            "HumanISP 대비 universal superiority",
            "PascalRAW에서는 corrected gate -1 runs가 mAP50-95를 개선했지만 recall tradeoff가 남아 있고 전체 조건 superiority는 아닙니다.",
            "multi-seed, matched recipe, hard/adverse held-out gate에서 같은 FP budget 또는 same recall budget으로 비교.",
            "Medium only for hard/adverse slices, low for easy whole-val.",
        ),
        (
            "All CFA utility",
            "현재 강한 native evidence는 특정 source CFA 중심입니다.",
            "RGGB/GRBG/BGGR/GBRG별 native 또는 credible remosaic sweep, same sensor protocol.",
            "Medium.",
        ),
        (
            "LensPSF-aware production value",
            "작은 PSF에서는 효과가 거의 안 보이는 것이 맞고, 지금은 diagnostic 수준입니다.",
            "의미 있는 PSF blur 조건에서 object-boundary confidence와 detector FP/recall을 비교.",
            "Medium if PSF is strong enough.",
        ),
        (
            "RGB+Aux production DNN gain",
            "Compact DNN ablation은 통과했고 YOLO corrected runs도 초기 양수 신호가 있으나, 아직 PascalRAW small native split 수준이며 seed3/seed4 repeat가 독립 variance 검증은 아닙니다.",
            "Matched RGB-only vs RGB+Aux detector with adequate data, hard-slice APs/APm/APl, low-light split.",
            "Unknown-medium; data volume and recipe matter.",
        ),
        (
            "Earlier YOLO aux-stem learned-branch claim",
            "최근 확인 결과 일부 이전 aux-stem run은 stem이 optimizer 생성 뒤에 붙어 gate/aux branch가 거의 학습되지 않았을 가능성이 있습니다.",
            "optimizer-before-stem fix가 들어간 wrapper로 matched RGB-only/RGB+Aux 실험을 다시 돌리고 gate movement와 hard-slice metric을 함께 확인.",
            "Medium after corrected matched run; low for old aux-stem runs as standalone evidence.",
        ),
        (
            "Adverse-condition RAW advantage",
            "LOD/AODRaw가 아직 로컬 평가 가능한 상태가 아닙니다.",
            "LOD complete download, archive/native format audit, low-light subset smoke, then full held-out gate.",
            "Medium-high if low-light edges/noise artifacts are present.",
        ),
        (
            "Object boundary edge confidence",
            "현재 scene-edge proxy는 object GT boundary와 다릅니다.",
            "GT box/mask boundary 주변 HumanISP/PerceptionISP edge confidence alignment metric 추가.",
            "Medium-high for small/thin object analysis.",
        ),
    ]


def _roadmap_rows() -> list[tuple[str, str, str, str]]:
    return [
        ("P0", "Make the corrected-path variance protocol genuinely stochastic", "Different split/shuffle/augmentation/initialization plus gate movement, freeze state, full-val metrics, hard-slice AP.", "Keep only if it beats RGB continued control across genuinely independent seeds or hard/adverse slices."),
        ("P0", "Keep corrected RGB+Aux aux-stem as the only valid learned-branch path", "setup_model-before-optimizer flag, freeze state, final gate state.", "Use only runs created after setup_model-before-optimizer fix."),
        ("P0", "Stop plain hard oversampling-only direction", "Hardx5 report already shows AP/localization degradation.", "Change objective/architecture rather than only repeating hard samples."),
        ("P0", "Complete LOD download and inspect archive", "native/raw-like format, image count, annotations, decode smoke.", "Proceed if images and labels align."),
        ("P1", "Mine hard PascalRAW cases", "small/thin/weak-edge/blur/artifact slice list and casebook.", "Train/evaluate on hard slices without overfitting full-val."),
        ("P1", "Aux pretraining refinement", "RGB feature matching for aux branch and gated_sum stability comparison.", "Keep if it improves convergence or hard-slice AP."),
        ("P1", "Object-boundary edge metric", "HumanISP RGB edge, PerceptionISP RGB edge, aux edge vs GT boundary confidence.", "Use for sensor/ISP co-design claim."),
        ("P2", "CFA/LensPSF sweep", "CFA-by-CFA and PSF-strength utility matrix.", "Use for joint sensor/ISP design story."),
        ("P2", "AODRaw or larger adverse benchmark", "night/rain/fog/glare/HDR condition-wise AP.", "Use only after disk/access are stable."),
    ]


def _source_rows(paths: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    rows = []
    for name, path in paths.items():
        path_str = str(path)
        report = path_str
        if path_str.endswith("/summary.json"):
            report = path_str[: -len("/summary.json")] + "/index.html"
        elif path_str.endswith("_summary.json"):
            report = str(Path(path_str).with_name("index.html"))
        rows.append((name, path_str, report))
    return rows


def _native_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    metric = _map(summary.get("metric_summary"))
    deltas = _map(metric.get("deltas"))
    provenance = _map(summary.get("provenance"))
    return {
        "status": summary.get("status", "missing"),
        "claim_status": summary.get("claim_status", "unknown"),
        "sample_count": summary.get("sample_count"),
        "precision50_delta": deltas.get("precision@0.50_mean"),
        "fp50_delta": deltas.get("fp@0.50_mean"),
        "recall50_delta": deltas.get("recall@0.50_mean"),
        "native_raw_fraction": provenance.get("native_raw_source_accepted_fraction"),
        "true_cfa_fraction": provenance.get("true_sensor_cfa_mosaic_fraction"),
        "source_cfa_patterns": provenance.get("source_cfa_patterns"),
        "boundary": summary.get("claim_boundary", ""),
    }


def _dense_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    means = _map(summary.get("mean_by_mode"))
    deltas = _map(summary.get("deltas_vs_none"))
    full = _map(means.get("none"))
    zero_aux = _map(deltas.get("zero_aux"))
    zero_rgb = _map(deltas.get("zero_rgb"))
    shuffle_aux = _map(deltas.get("shuffle_aux"))
    return {
        "status": summary.get("status", "missing"),
        "claim_status": summary.get("claim_status", "unknown"),
        "seed_count": summary.get("seed_count"),
        "test_sample_count": summary.get("test_sample_count"),
        "full_precision": full.get("precision"),
        "full_recall": full.get("recall"),
        "full_fp": full.get("fp"),
        "zero_aux_recall_delta": zero_aux.get("recall"),
        "zero_aux_precision_delta": zero_aux.get("precision"),
        "zero_rgb_recall_delta": zero_rgb.get("recall"),
        "shuffle_aux_recall_delta": shuffle_aux.get("recall"),
        "boundary": summary.get("claim_boundary", ""),
    }


def _scene_metrics(edge_summary: Mapping[str, Any], aux_summary: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _map(edge_summary.get("aggregate"))
    best = _map(aux_summary.get("best_candidate"))
    return {
        "status": edge_summary.get("status", "missing"),
        "sample_count": edge_summary.get("sample_count"),
        "perception_rgb_d_f1": aggregate.get("perception_rgb_minus_human_source_edge_f1_mean"),
        "perception_rgb_win_rate": aggregate.get("perception_rgb_source_edge_f1_win_rate"),
        "aux_strength_d_f1": aggregate.get("perception_aux_strength_minus_human_source_edge_f1_mean"),
        "aux_strength_win_rate": aggregate.get("perception_aux_strength_source_edge_f1_win_rate"),
        "aux_confidence_d_f1": aggregate.get("perception_aux_confidence_minus_human_source_edge_f1_mean"),
        "aux_best_name": best.get("name"),
        "aux_best_d_f1": best.get("minus_human_source_edge_f1_mean"),
    }


def _read_train_summary(root: Path, path: Path) -> dict[str, Any]:
    summary_path = root / path
    if summary_path.is_file():
        summary = _read_json(summary_path)
        return {
            "status": "complete",
            "path": str(summary_path),
            "config": {
                "epochs": summary.get("epochs"),
                "imgsz": summary.get("imgsz"),
                "batch": summary.get("batch"),
                "seed": summary.get("seed"),
                "device": summary.get("device"),
                "name": summary.get("name"),
                "train_overrides": summary.get("train_overrides"),
            },
            "metrics": _train_metrics(summary),
            "aux_stem": summary.get("aux_stem"),
            "aux_stem_state_after_train": summary.get("aux_stem_state_after_train"),
            "aux_adapter": summary.get("aux_feature_adapter_init"),
        }
    run_dir = summary_path.parent
    status = "running" if run_dir.is_dir() else "missing"
    return {"status": status, "path": str(summary_path), "metrics": {}, "config": {"name": run_dir.name}}


def _train_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    results = _map(summary.get("results_dict"))
    return {
        "precision": _float(results.get("metrics/precision(B)")),
        "recall": _float(results.get("metrics/recall(B)")),
        "map50": _float(results.get("metrics/mAP50(B)")),
        "map5095": _float(results.get("metrics/mAP50-95(B)")),
    }


def _yolo_train_comparison(base: Mapping[str, Any], candidate: Mapping[str, Any], label: str) -> dict[str, Any]:
    base_metrics = {"status": base.get("status", "unknown"), **_map(base.get("metrics"))}
    candidate_metrics = {"status": candidate.get("status", "unknown"), **_map(candidate.get("metrics"))}
    if base.get("status") != "complete" or candidate.get("status") != "complete":
        statuses = {str(base.get("status", "unknown")), str(candidate.get("status", "unknown"))}
        status = "running" if "running" in statuses else "missing" if "missing" in statuses else "partial"
        return {
            "status": status,
            "comparison": label,
            "base": base_metrics,
            "candidate": candidate_metrics,
            "candidate_aux_stem": candidate.get("aux_stem"),
            "candidate_aux_stem_state_after_train": candidate.get("aux_stem_state_after_train"),
            "message": "Final training summaries are not both available yet.",
        }
    deltas = {
        key: _float(candidate_metrics.get(key)) - _float(base_metrics.get(key))
        for key in ("precision", "recall", "map50", "map5095")
    }
    return {
        "status": "partial" if deltas["map5095"] < 0 else "pass",
        "comparison": label,
        "base": base_metrics,
        "candidate": candidate_metrics,
        "candidate_aux_stem": candidate.get("aux_stem"),
        "candidate_aux_stem_state_after_train": candidate.get("aux_stem_state_after_train"),
        "precision_delta": deltas["precision"],
        "recall_delta": deltas["recall"],
        "map50_delta": deltas["map50"],
        "map5095_delta": deltas["map5095"],
        "delta_text": f"dR={_fmt(deltas['recall'], signed=True)}, dAP50-95={_fmt(deltas['map5095'], signed=True)}",
        "interpretation": _yolo_interpretation(deltas),
    }


def _optimizer_fix_smoke(smoke: Mapping[str, Any]) -> dict[str, Any]:
    if smoke.get("status") != "complete":
        return {
            "status": smoke.get("status", "missing"),
            "message": "Optimizer/freeze smoke summary is not available yet.",
        }
    stem = _map(smoke.get("aux_stem"))
    state = _map(smoke.get("aux_stem_state_after_train"))
    freeze = _map(stem.get("freeze_enforced_after_ultralytics_freeze"))
    setup_ok = stem.get("setup_phase") == "setup_model_before_optimizer"
    freeze_ok = freeze.get("status") == "frozen" and not any(bool(v) for v in _list(freeze.get("requires_grad_after")))
    gate_present = state.get("status") == "present" and state.get("gate_sigmoid_mean") is not None
    status = "pass" if setup_ok and freeze_ok and gate_present else "partial"
    return {
        "status": status,
        "message": (
            "stem before optimizer; freeze re-enforced; gate state recorded"
            if status == "pass"
            else "optimizer/freeze path still needs inspection"
        ),
        "setup_phase": stem.get("setup_phase", "unknown"),
        "freeze_status": freeze.get("status", "unknown"),
        "gate_sigmoid_mean": state.get("gate_sigmoid_mean"),
        "rgb_branch_requires_grad": state.get("rgb_branch_requires_grad"),
        "interpretation": (
            "This validates the training path used for the next real matched experiment; the smoke metrics are not a performance claim."
            if status == "pass"
            else "Do not use aux-stem performance claims until this path is verified."
        ),
    }


def _hard_pair(summary: Mapping[str, Any], base_name: str, candidate_name: str, label: str) -> dict[str, Any]:
    runs = _map(summary.get("runs"))
    base = _map(runs.get(base_name))
    candidate = _map(runs.get(candidate_name))
    if not base or not candidate:
        return {"status": "missing", "label": label, "filters": {}}
    filters = {}
    for filter_name in ("all", "small", "thin", "small_or_thin"):
        base_filter = _map(_map(base.get("filters")).get(filter_name))
        cand_filter = _map(_map(candidate.get("filters")).get(filter_name))
        filters[filter_name] = _hard_filter_delta(base_filter, cand_filter)
    all_filter = _map(filters.get("all"))
    small_filter = _map(filters.get("small"))
    small_thin_filter = _map(filters.get("small_or_thin"))
    status = (
        "pass"
        if _float(all_filter.get("map50_delta")) >= 0
        and _float(all_filter.get("map5095_delta")) >= 0
        and _float(small_filter.get("map5095_delta")) >= 0
        and _float(small_thin_filter.get("map50_delta")) >= 0
        and _float(small_thin_filter.get("map5095_delta")) >= 0
        else "partial"
    )
    return {
        "status": status,
        "label": label,
        "base": base_name,
        "candidate": candidate_name,
        "filters": filters,
    }


def _hard_candidate_pairs(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = [
        ("edge6_adapter512_seed1 - rgb512_seed1", "rgb512_seed1", "edge6_adapter512_seed1", "partially matched", "No; adapter only, use as supporting context."),
        ("edge6_gated512_seed1 - rgb512_seed1", "rgb512_seed1", "edge6_gated512_seed1", "mismatched recipe", "No; promising candidate only."),
        ("edge6_gated640_seed1 - rgb640_seed1", "rgb640_seed1", "edge6_gated640_seed1", "closer but verify config", "Only after config audit."),
        ("edge6_gated_hardx2_seed1 - rgb640_hardx2_seed1", "rgb640_hardx2_seed1", "edge6_gated_hardx2_seed1", "hard-mined pair", "Candidate; needs matched seed repeat."),
        ("edge6_gated_hardx3_seed1 - rgb640_hardx3_seed1", "rgb640_hardx3_seed1", "edge6_gated_hardx3_seed1", "hard-mined pair", "Candidate; needs matched seed repeat."),
        ("edge6_gated_from_rgb_hardx3_seed1 - rgb640_hardx3_seed1", "rgb640_hardx3_seed1", "edge6_gated_from_rgb_hardx3_seed1", "hard-mined aux-init pair", "Candidate; needs matched seed repeat."),
    ]
    rows: list[dict[str, Any]] = []
    for label, base, candidate, fairness, claim_use in pairs:
        row = _hard_pair(summary, base, candidate, label)
        if row.get("status") != "missing":
            row["fairness"] = fairness
            row["claim_use"] = claim_use
            rows.append(row)
    return rows


def _hard_filter_delta(base_filter: Mapping[str, Any], cand_filter: Mapping[str, Any]) -> dict[str, Any]:
    base_fixed = _map(_map(base_filter.get("fixed_conf")).get("conf_0.25_iou_0.50"))
    cand_fixed = _map(_map(cand_filter.get("fixed_conf")).get("conf_0.25_iou_0.50"))
    return {
        "target_gt_count": cand_filter.get("target_gt_count", base_filter.get("target_gt_count")),
        "map50_delta": _float(cand_filter.get("mAP50")) - _float(base_filter.get("mAP50")),
        "map5095_delta": _float(cand_filter.get("mAP50_95")) - _float(base_filter.get("mAP50_95")),
        "precision025_delta": _float(cand_fixed.get("precision")) - _float(base_fixed.get("precision")),
        "recall025_delta": _float(cand_fixed.get("recall")) - _float(base_fixed.get("recall")),
    }


def _raw_acquisition_state(raw: Mapping[str, Any], lod_ready: Mapping[str, Any], lod: Mapping[str, Any]) -> dict[str, Any]:
    local = _map(raw.get("local_state"))
    annotations = bool(local.get("aodraw_annotations_present"))
    raw_zip = _map(local.get("aodraw_test_raw_zip")).get("status", "unknown")
    srgb_zip = _map(local.get("aodraw_srgb_zip")).get("status", "unknown")
    return {
        "lod_status": lod.get("status", "unknown"),
        "lod_message": lod.get("message", ""),
        "lod_readiness": lod_ready.get("status", "unknown"),
        "aodraw_state": f"annotations_present={annotations}, test_raw={raw_zip}, srgb={srgb_zip}",
        "status": "partial" if lod.get("status") in {"partial", "present"} else "missing",
    }


def _lod_download_status(root: Path) -> dict[str, Any]:
    candidates = [
        root / "data/raw_datasets/lod/downloads/LOD_BMVC2021.zip",
        root / "data/raw_datasets/lod/downloads/LOD_BMVC2021.zip.part",
        root / "data/raw_datasets/lod/downloads/LOD_BMVC21.zip",
        root / "data/raw_datasets/lod/downloads/LOD_BMVC21.zip.part",
    ]
    for path in candidates:
        if path.is_file():
            size = int(path.stat().st_size)
            ratio = size / float(LOD_EXPECTED_BYTES)
            status = "present" if ratio >= 0.95 and not path.name.endswith(".part") else "partial"
            return {
                "status": status,
                "path": str(path),
                "size_bytes": size,
                "expected_bytes": LOD_EXPECTED_BYTES,
                "progress": ratio,
                "message": f"{path.name}: {size / 1024**3:.2f} GiB / {LOD_EXPECTED_BYTES / 1024**3:.2f} GiB ({ratio * 100:.1f}%)",
            }
    return {"status": "missing", "path": "", "size_bytes": 0, "expected_bytes": LOD_EXPECTED_BYTES, "progress": 0.0, "message": "LOD archive not found locally."}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "missing_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _overall_status(*sections: Mapping[str, Any]) -> str:
    statuses = {str(section.get("status", "unknown")) for section in sections}
    if "fail" in statuses or "blocked" in statuses:
        return "partial"
    if "missing" in statuses or "running" in statuses or "partial" in statuses:
        return "partial"
    if statuses and statuses <= {"pass", "supported", "present"}:
        return "pass"
    return "partial"


def _cards(cards: Sequence[Sequence[Any]]) -> str:
    chunks = []
    for label, value, detail, status in cards:
        status_text = _esc(status)
        chunks.append(
            f'<article class="card"><div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(value)}</div>'
            f'<div class="detail">{_esc(detail)}</div>'
            f'<p><span class="status {status_text}">{status_text}</span></p></article>'
        )
    return "".join(chunks)


def _tab(tab_id: str, label: str, active: bool) -> str:
    cls = "tab-button active" if active else "tab-button"
    return f'<button class="{cls}" type="button" data-tab="{_esc(tab_id)}">{_esc(label)}</button>'


def _rows(rows: Sequence[Sequence[Any]], headers: Sequence[str] = ("Item", "Evidence")) -> str:
    head = "".join(f"<th>{_esc(item)}</th>" for item in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_esc(item)}</td>" for item in row) + "</tr>")
    return f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"


def _metric_text(metrics: Any) -> str:
    data = _map(metrics)
    return (
        f"P={_fmt(data.get('precision'))}, R={_fmt(data.get('recall'))}, "
        f"mAP50={_fmt(data.get('map50'))}, mAP50-95={_fmt(data.get('map5095'))}"
    )


def _hard_card_text(hard: Mapping[str, Any]) -> str:
    filters = _map(hard.get("filters"))
    small_thin = _map(filters.get("small_or_thin"))
    return (
        f"small|thin dAP50={_fmt(small_thin.get('map50_delta'), signed=True)}, "
        f"dAP50-95={_fmt(small_thin.get('map5095_delta'), signed=True)}"
    )


def _hard_claim_text(hard: Mapping[str, Any]) -> str:
    filters = _map(hard.get("filters"))
    small_thin = _map(filters.get("small_or_thin"))
    if not small_thin:
        return "Hard-slice summary missing."
    label = str(hard.get("label", "edge6_seed2 - rgb_seed2")).split(" - ")[0]
    return (
        f"{label} small|thin dAP50={_fmt(small_thin.get('map50_delta'), signed=True)}, "
        f"dAP50-95={_fmt(small_thin.get('map5095_delta'), signed=True)}."
    )


def _hard_interpretation(row: Mapping[str, Any]) -> str:
    map50 = _float(row.get("map50_delta"))
    map5095 = _float(row.get("map5095_delta"))
    recall = _float(row.get("recall025_delta"))
    if map50 > 0 and map5095 > 0:
        return "AP improves; usable hard-slice evidence."
    if recall > 0 and (map50 < 0 or map5095 < 0):
        return "Recall signal but AP/localization tradeoff."
    if map50 < 0 and map5095 < 0:
        return "Plain edge6 is weaker than RGB on this slice."
    return "Mixed; needs gated or hard-mined validation."


def _yolo_interpretation(deltas: Mapping[str, float]) -> str:
    if _float(deltas.get("map5095")) > 0 and _float(deltas.get("recall")) >= 0:
        return "Matched RGB+Aux improves AP quality and does not lose recall."
    if _float(deltas.get("recall")) > 0 and _float(deltas.get("map5095")) < 0:
        return "Aux path improves recall but loses AP50-95; this is a signal path, not broad superiority."
    return "No clear detector advantage over RGB-only under this recipe."


def _fmt(value: Any, *, signed: bool = False, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number:.{digits}f}"


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _esc(value: Any) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
