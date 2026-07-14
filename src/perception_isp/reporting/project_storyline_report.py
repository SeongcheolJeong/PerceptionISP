"""Build a standalone tabbed PerceptionISP engineering storyline report."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_PATHS = {
    "claim_dashboard": Path("reports/perception_claim_dashboard_pascalraw_native1581_rgb_aux_3seed_learned_evidence_v1/claim_dashboard_summary.json"),
    "native_audit": Path("reports/perception_native_heldout_benchmark_audit_pascalraw_rawdet7_val750_fusion_v2/native_heldout_benchmark_audit_summary.json"),
    "dense_ablation": Path("reports/perception_dense_input_ablation_gate_pascalraw_native1581_union_seed101_202_303_broad_cached_v1/dense_input_ablation_summary.json"),
    "scene_edge": Path("reports/perception_scene_edge_confidence_bus_cfa_psf_sweep/scene_edge_confidence_summary.json"),
    "scene_edge_aux": Path("reports/perception_scene_edge_aux_sweep_sid_lowlight_native16_edge_evidence_v1/scene_edge_aux_sweep_summary.json"),
    "yolo_edge6_audit": Path("reports/perception_yolo_edge6_evidence_audit_current_v1/yolo_evidence_audit_summary.json"),
    "raw_acquisition": Path("reports/perception_raw_dataset_acquisition_lod_priority_v1/raw_dataset_acquisition_summary.json"),
    "lod_readiness": Path("reports/perception_lod_local_readiness_current_v1/lod_local_readiness_summary.json"),
}

YOLO_MATCHED_RUNS = {
    "rgb_only_seed2": Path("runs/detect/outputs/yolo_aux_pascalraw1581/rgb_only3_full_e20_mps_seed2_adamw_lr5e4_v1/perception_yolo_aux_train_summary.json"),
    "rgb_aux_edge6_seed2": Path("runs/detect/outputs/yolo_aux_pascalraw1581/rgb_aux_edge6_adapter025_full_e20_mps_seed2_adamw_lr5e4_v1/perception_yolo_aux_train_summary.json"),
}

LOD_EXPECTED_BYTES = 22_000_000_000


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the current PerceptionISP tabbed storyline HTML report.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--output-dir", default="reports/perception_project_storyline_tabs_current_v1")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    summary = build_storyline_summary(root)
    html_path = write_storyline_report(summary, root / args.output_dir)
    print(json.dumps({"report": str(html_path), "summary_json": str(html_path.with_name("project_storyline_summary.json"))}, indent=2))
    return 0


def build_storyline_summary(root: Path) -> dict[str, Any]:
    evidence = {name: _read_json(root / path) for name, path in SUMMARY_PATHS.items()}
    native = evidence["native_audit"]
    dense = evidence["dense_ablation"]
    scene = evidence["scene_edge"]
    aux_sweep = evidence["scene_edge_aux"]
    yolo_audit = evidence["yolo_edge6_audit"]
    claim_dashboard = evidence["claim_dashboard"]
    yolo_seed2 = _matched_yolo_seed2(root)

    native_metric = _mapping(native.get("metric_summary"))
    native_delta = _mapping(native_metric.get("deltas"))
    native_prov = _mapping(native.get("provenance"))
    dense_deltas = _mapping(dense.get("deltas_vs_none"))
    dense_means = _mapping(dense.get("mean_by_mode"))
    scene_agg = _mapping(scene.get("aggregate"))
    aux_best = _mapping(aux_sweep.get("best_candidate"))

    cards = [
        {
            "label": "Native RAW benchmark",
            "value": f"{_int(native.get('sample_count'))} samples",
            "detail": (
                f"dP50={_fmt(native_delta.get('precision@0.50_mean'), signed=True)}, "
                f"dFP50={_fmt(native_delta.get('fp@0.50_mean'), signed=True)}, "
                f"dR50={_fmt(native_delta.get('recall@0.50_mean'), signed=True)}"
            ),
            "status": str(native.get("status", "unknown")),
        },
        {
            "label": "Aux input dependence",
            "value": f"{_int(dense.get('seed_count'))} seeds",
            "detail": f"zero-aux recall delta={_fmt(_mapping(dense_deltas.get('zero_aux')).get('recall'), signed=True)}",
            "status": str(dense.get("status", "unknown")),
        },
        {
            "label": "Scene-edge evidence",
            "value": f"{_int(scene.get('sample_count')) or 12} cases",
            "detail": (
                f"Perception RGB dF1={_fmt(scene_agg.get('perception_rgb_minus_human_source_edge_f1_mean'), signed=True)}, "
                f"aux strength dF1={_fmt(scene_agg.get('perception_aux_strength_minus_human_source_edge_f1_mean'), signed=True)}"
            ),
            "status": str(scene.get("status", "unknown")),
        },
        {
            "label": "LOD acquisition",
            "value": _lod_download_status(root)["status"],
            "detail": _lod_download_status(root)["message"],
            "status": _lod_download_status(root)["status"],
        },
        {
            "label": "YOLO edge6 audit",
            "value": f"{_int(yolo_audit.get('fair_valid_edge6_rgb_comparison_count'))} fair",
            "detail": (
                f"{_int(yolo_audit.get('valid_edge6_run_count'))} valid runs, "
                f"{_int(yolo_audit.get('training_mismatched_valid_edge6_rgb_comparison_count'))} mismatched rows"
            ),
            "status": str(yolo_audit.get("status", "unknown")),
        },
        {
            "label": "Matched YOLO seed2",
            "value": "edge6 vs RGB",
            "detail": (
                f"dR={_fmt(_mapping(yolo_seed2.get('deltas')).get('recall'), signed=True)}, "
                f"dAP50={_fmt(_mapping(yolo_seed2.get('deltas')).get('mAP50'), signed=True)}, "
                f"dAP50-95={_fmt(_mapping(yolo_seed2.get('deltas')).get('mAP50_95'), signed=True)}"
            ),
            "status": str(yolo_seed2.get("status", "unknown")),
        },
    ]

    supported_claims = list(_mapping(claim_dashboard).get("decisions") or [])
    if not supported_claims:
        supported_claims = [
            {
                "status": "supported",
                "claim": "PascalRAW native held-out gate shows FP/precision improvement with bounded recall loss for RGGB native RAW.",
            },
            {
                "status": "supported",
                "claim": "RGB+Aux compact DNN input ablation shows that the trained model is using aux channels at the selected operating point.",
            },
        ]

    return {
        "name": "PerceptionISP engineering storyline report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "status": _overall_status(native, dense, scene),
        "cards": cards,
        "evidence_paths": {name: str(path) for name, path in SUMMARY_PATHS.items()},
        "supported_claims": supported_claims,
        "metrics": {
            "native": {
                "status": native.get("status"),
                "claim_status": native.get("claim_status"),
                "sample_count": native.get("sample_count"),
                "precision_delta_50": native_delta.get("precision@0.50_mean"),
                "recall_delta_50": native_delta.get("recall@0.50_mean"),
                "fp_delta_50": native_delta.get("fp@0.50_mean"),
                "native_raw_fraction": native_prov.get("native_raw_source_accepted_fraction"),
                "true_cfa_fraction": native_prov.get("true_sensor_cfa_mosaic_fraction"),
                "pattern_remapped_count": native_prov.get("pattern_remapped_count"),
                "source_cfa_patterns": native_prov.get("source_cfa_patterns"),
            },
            "dense_ablation": {
                "status": dense.get("status"),
                "claim_status": dense.get("claim_status"),
                "seed_count": dense.get("seed_count"),
                "test_sample_count": dense.get("test_sample_count"),
                "mean_by_mode": dense_means,
                "deltas_vs_none": dense_deltas,
            },
            "scene_edge": {
                "status": scene.get("status"),
                "aggregate": scene_agg,
                "aux_best_candidate": aux_best,
            },
            "yolo_edge6_audit": {
                "status": yolo_audit.get("status"),
                "valid_edge6_run_count": yolo_audit.get("valid_edge6_run_count"),
                "edge6_name_without_edge_evidence_count": yolo_audit.get("edge6_name_without_edge_evidence_count"),
                "valid_edge6_rgb_comparison_count": yolo_audit.get("valid_edge6_rgb_comparison_count"),
                "fair_valid_edge6_rgb_comparison_count": yolo_audit.get("fair_valid_edge6_rgb_comparison_count"),
                "training_mismatched_valid_edge6_rgb_comparison_count": yolo_audit.get("training_mismatched_valid_edge6_rgb_comparison_count"),
                "valid_edge6_all_filter_mAP50_win_count": yolo_audit.get("valid_edge6_all_filter_mAP50_win_count"),
                "valid_edge6_hard_filter_mAP50_win_count": yolo_audit.get("valid_edge6_hard_filter_mAP50_win_count"),
                "interpretation": yolo_audit.get("interpretation"),
                "next_actions": yolo_audit.get("next_actions"),
            },
            "yolo_seed2": yolo_seed2,
            "lod_download": _lod_download_status(root),
        },
        "claim_boundaries": [
            str(native.get("claim_boundary", "")),
            str(dense.get("claim_boundary", "")),
            str(scene.get("claim_boundary", "")),
            str(aux_sweep.get("claim_boundary", "")),
            str(yolo_audit.get("claim_boundary", "")),
        ],
        "raw_acquisition": evidence["raw_acquisition"],
        "lod_readiness": evidence["lod_readiness"],
    }


def write_storyline_report(summary: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "project_storyline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = output_dir / "index.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    return html_path


def _render_html(summary: Mapping[str, Any]) -> str:
    cards = "\n".join(_card(row) for row in _list(summary.get("cards")))
    metrics = _mapping(summary.get("metrics"))
    native = _mapping(metrics.get("native"))
    dense = _mapping(metrics.get("dense_ablation"))
    scene = _mapping(metrics.get("scene_edge"))
    yolo = _mapping(metrics.get("yolo_edge6_audit"))
    yolo_seed2 = _mapping(metrics.get("yolo_seed2"))
    lod = _mapping(metrics.get("lod_download"))
    raw = _mapping(summary.get("raw_acquisition"))
    lod_ready = _mapping(summary.get("lod_readiness"))
    source_links = _source_links(_mapping(summary.get("evidence_paths")))
    claim_ladder_rows = _claim_ladder_rows(native, dense, scene, yolo, lod, yolo_seed2)
    purpose_rows = _purpose_rows()
    expected_effect_rows = _expected_effect_rows(yolo_seed2)
    dataset_test_rows = _dataset_test_rows(native, dense, scene, yolo, lod, raw, lod_ready, yolo_seed2)
    expectation_rows = _expectation_rows(yolo_seed2)

    native_rows = [
        ("Dataset/protocol", "PascalRAW native RAW, 750 held-out images, true sensor CFA mosaic, source/target CFA matched."),
        ("CFA provenance", f"source={_json(native.get('source_cfa_patterns'))}, remapped={_int(native.get('pattern_remapped_count'))}"),
        ("Detector effect", f"precision@0.50 delta {_fmt(native.get('precision_delta_50'), signed=True)}, FP@0.50 delta {_fmt(native.get('fp_delta_50'), signed=True)}, recall@0.50 delta {_fmt(native.get('recall_delta_50'), signed=True)}"),
        ("Claim boundary", "This supports a native RGGB FP-reduction / precision-gain claim with bounded recall loss. It does not prove all CFA patterns, all adverse conditions, or production detector superiority."),
    ]
    dense_rows = _dense_rows(dense)
    scene_rows = _scene_rows(scene)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PerceptionISP Engineering Storyline</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --paper: #ffffff;
      --ink: #17202a;
      --muted: #5c6b7a;
      --line: #d9e0e8;
      --blue: #1f6feb;
      --green: #14804a;
      --amber: #9a6700;
      --red: #b42318;
      --slate: #263445;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    header {{ padding: 30px 34px 22px; background: var(--paper); border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 22px 0 10px; font-size: 17px; }}
    p {{ line-height: 1.55; }}
    code {{ background: #eef2f7; border-radius: 4px; padding: 1px 5px; }}
    .subtitle {{ margin: 0; color: var(--muted); max-width: 1180px; line-height: 1.55; }}
    .wrap {{ padding: 18px 34px 40px; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 120px; }}
    .card .label {{ color: var(--muted); font-size: 13px; }}
    .card .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .card .detail {{ color: var(--muted); margin-top: 8px; line-height: 1.45; }}
    .status {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid var(--line); background: #eef2f7; }}
    .pass, .supported, .present {{ color: var(--green); border-color: #b7e3c7; background: #eefaf2; }}
    .partial, .blocked, .unknown {{ color: var(--amber); border-color: #ead18a; background: #fff8df; }}
    .fail, .missing {{ color: var(--red); border-color: #f1b8b3; background: #fff1f0; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; border-bottom: 1px solid var(--line); margin-top: 20px; }}
    .tab-button {{ border: 1px solid var(--line); border-bottom: 0; background: #eef2f7; color: var(--slate); padding: 10px 13px; border-radius: 8px 8px 0 0; cursor: pointer; font-weight: 650; }}
    .tab-button.active {{ background: var(--paper); color: var(--blue); }}
    .tab-panel {{ display: none; background: var(--paper); border: 1px solid var(--line); border-top: 0; padding: 20px; border-radius: 0 0 8px 8px; }}
    .tab-panel.active {{ display: block; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .flow {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 14px 0 4px; }}
    .node {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfcfe; min-height: 108px; }}
    .node strong {{ display: block; margin-bottom: 7px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid var(--line); padding: 9px 10px; vertical-align: top; text-align: left; }}
    th {{ background: #f0f4f9; }}
    .note {{ border-left: 4px solid var(--blue); background: #f2f7ff; padding: 12px 14px; margin: 14px 0; }}
    .warn {{ border-left-color: var(--amber); background: #fff9e8; }}
    .bad {{ border-left-color: var(--red); background: #fff2f1; }}
    .small {{ color: var(--muted); font-size: 13px; }}
    ul {{ line-height: 1.55; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 1100px) {{ .cards, .grid2, .flow {{ grid-template-columns: 1fr; }} .wrap, header {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
<header>
  <h1>PerceptionISP Engineering Storyline</h1>
  <p class="subtitle">목적, 구현 아키텍처, 데이터셋/테스트, 현재 evidence, 아직 검증하지 못한 부분과 다음 검증 순서를 한 문서로 묶은 standalone report입니다. 이 문서는 “자랑 가능한 근거”와 “아직 주장하면 안 되는 경계”를 분리해서 보여줍니다.</p>
  <p class="small">Generated: {html_lib.escape(str(summary.get('generated_at', '')))} · Overall: <span class="status {html_lib.escape(str(summary.get('status', 'unknown')))}">{html_lib.escape(str(summary.get('status', 'unknown')))}</span></p>
</header>
<div class="wrap">
  <section class="cards">{cards}</section>
  <nav class="tabs" aria-label="Report tabs">
    {_tab_button("overview", "1. Storyline", True)}
    {_tab_button("architecture", "2. Architecture", False)}
    {_tab_button("implemented", "3. Implemented Blocks", False)}
    {_tab_button("evidence", "4. Evidence", False)}
    {_tab_button("interpretation", "5. Interpretation", False)}
    {_tab_button("gaps", "6. Gaps", False)}
    {_tab_button("roadmap", "7. Next Work", False)}
    {_tab_button("sources", "8. Sources", False)}
  </nav>

  <section id="overview" class="tab-panel active">
    <h2>Storyline</h2>
    <p>HumanISP는 사람이 보기 좋은 RGB를 만드는 방향으로 설계됩니다. PerceptionISP는 같은 RAW에서 detector가 사용할 수 있는 edge, blur, noise, demosaic reliability, PSF-aware confidence 같은 신호를 함께 만들어서, RGB-only detector가 놓치거나 헷갈리는 조건을 보조하려는 front-end입니다.</p>
    <div class="note">현재 가장 방어 가능한 주장은 “PerceptionISP fusion 또는 RGB+Aux 경로가 일부 native RAW held-out 조건에서 FP를 줄이고 precision을 올릴 수 있으며, compact RGB+Aux DNN은 aux channel을 실제 입력 신호로 사용한다”입니다. “모든 조건에서 HumanISP보다 성능이 좋다” 또는 “production detector까지 검증됐다”는 아직 이릅니다.</div>
    <h3>Purpose-to-evidence chain</h3>
    <table>{_kv_rows(purpose_rows, headers=("Engineering question", "Why PerceptionISP was built", "Current answer"))}</table>
    <h3>Claim ladder</h3>
    <table>{_kv_rows(claim_ladder_rows, headers=("Claim level", "Current status", "Evidence", "What would make it stronger"))}</table>
    <h3>Why this matters</h3>
    <ul>
      <li>센서/CFA/LensPSF/ISP가 perception 결과에 주는 영향을 실제 실험 항목으로 분리했습니다.</li>
      <li>Aux map을 독립 detector로 쓰는 대신 RGB detector class label을 유지하고 aux evidence를 calibration/training input으로 쓰는 방향을 정리했습니다.</li>
      <li>PascalRAW native RAW에서 더 큰 held-out gate를 만들었고, LOD/AODRaw 같은 adverse RAW dataset으로 확장할 acquisition path를 만들었습니다.</li>
    </ul>
  </section>

  <section id="architecture" class="tab-panel">
    <h2>Architecture</h2>
    <div class="flow">
      <div class="node"><strong>Scene / RAW Source</strong>CameraE2E RGB scene simulation, PascalRAW native NEF, KITTI/COCO proxy, SID/LOD/AODRaw acquisition targets.</div>
      <div class="node"><strong>Sensor Front-End</strong>CFA mosaic, source CFA tracking, LensPSF blur, noise and exposure stress. Pattern remapping is tracked and avoided for native claims.</div>
      <div class="node"><strong>HumanISP Branch</strong>Standard RGB-facing demosaic/tone path used as the comparison baseline for detector and edge metrics.</div>
      <div class="node"><strong>PerceptionISP Branch</strong>Edge-aware demosaic, denoise, artifact suppression, PSF-aware aux maps, edge evidence, reliability maps.</div>
      <div class="node"><strong>Perception Back-End</strong>Fusion/calibration gates, RGB+Aux export, compact DNN fine-tuning, input ablation, held-out claim dashboards.</div>
    </div>
    <h3>Design intent</h3>
    <p>PerceptionISP is not just another RGB rendering pipeline. The implementation keeps a DNN-facing tensor path: RGB channels plus aux channels such as edge strength, edge evidence, artifact likelihood, noise estimate, blur/PSF likelihood, and confidence maps. The strongest path now is not aux-only detection; it is RGB label/proposal semantics with aux evidence as a supporting signal or learned input.</p>
    <div class="note warn">중요한 판단: aux map 단독 detector는 object class label을 만들지 못하므로 label-aware mAP에서 거의 0점이 되는 구조입니다. 따라서 aux는 독립 detector가 아니라 RGB detector 또는 RGB+Aux DNN을 보조하는 evidence로 써야 합니다.</div>
    <h3>Expected effect by block</h3>
    <table>{_kv_rows(expected_effect_rows, headers=("Block", "Signal added by PerceptionISP", "Expected perception benefit", "Current validation level"))}</table>
  </section>

  <section id="implemented" class="tab-panel">
    <h2>Implemented Blocks</h2>
    <table>{_implemented_rows()}</table>
    <h3>Recent engineering fixes</h3>
    <ul>
      <li><code>edge6</code> export path를 추가해 <code>aux_edge_evidence</code>가 실제 base RGB+Aux tensor에 들어가도록 수정했습니다.</li>
      <li>YOLO aux dataset loader가 NPZ 내부의 실제 channel names를 읽도록 수정해 custom 6-channel tensor를 default stable6으로 오해하지 않게 했습니다.</li>
      <li>LOD acquisition/local-state support를 추가했고, 실제 Google Drive 파일명 <code>LOD_BMVC2021.zip</code> 및 partial 상태를 탐지하도록 보강했습니다.</li>
    </ul>
    <div class="note warn">구현됐다는 말은 “software pipeline에서 기능을 생성하고 테스트할 수 있다”는 의미입니다. 아직 모든 block이 production ISP 수준의 최적 품질, 모든 센서 조건, 모든 detector에서 검증됐다는 뜻은 아닙니다.</div>
  </section>

  <section id="evidence" class="tab-panel">
    <h2>Evidence</h2>
    <h3>Dataset and test matrix</h3>
    <table>{_kv_rows(dataset_test_rows, headers=("Dataset / test", "Purpose", "Current result", "Limitation"))}</table>
    <div class="grid2">
      <div>
        <h3>Native RAW detector gate</h3>
        <table>{_kv_rows(native_rows)}</table>
      </div>
      <div>
        <h3>RGB+Aux DNN input ablation</h3>
        <table>{_kv_rows(dense_rows)}</table>
      </div>
    </div>
    <h3>Scene edge and PSF evidence</h3>
    <table>{_kv_rows(scene_rows)}</table>
    <h3>YOLO edge6 evidence audit</h3>
    <table>{_kv_rows([
        ('Valid edge6 runs', yolo.get('valid_edge6_run_count', 0)),
        ('Valid edge6 vs RGB comparisons', yolo.get('valid_edge6_rgb_comparison_count', 0)),
        ('Fair valid comparisons', yolo.get('fair_valid_edge6_rgb_comparison_count', 0)),
        ('Training-mismatched valid rows', yolo.get('training_mismatched_valid_edge6_rgb_comparison_count', 0)),
        ('Fair all-filter mAP50 wins', yolo.get('valid_edge6_all_filter_mAP50_win_count', 0)),
        ('Fair hard-filter mAP50 wins', yolo.get('valid_edge6_hard_filter_mAP50_win_count', 0)),
        ('Flagged edge6-named runs', yolo.get('edge6_name_without_edge_evidence_count', 0)),
        ('Interpretation', yolo.get('interpretation', '')),
    ])}</table>
    <h3>Matched YOLO RGB-only vs RGB+Aux edge6 seed2</h3>
    <table>{_kv_rows(_yolo_seed2_rows(yolo_seed2))}</table>
    <h3>Dataset acquisition status</h3>
    <table>{_kv_rows([
        ('LOD download status', f"{lod.get('status', 'unknown')} - {lod.get('message', '')}"),
        ('LOD readiness', f"{lod_ready.get('status', 'unknown')} / ready_for_image_eval={lod_ready.get('ready_for_image_eval', False)}"),
        ('AODRaw local note', _aodraw_note(raw)),
        ('Recommended dataset decision', 'Use LOD first for practical low-light RAW validation; keep AODRaw as stronger but heavier adverse benchmark.'),
    ])}</table>
  </section>

  <section id="interpretation" class="tab-panel">
    <h2>Interpretation</h2>
    <h3>Supported now</h3>
    <ul>{_claim_list(_list(summary.get('supported_claims')))}</ul>
    <h3>What the numbers mean</h3>
    <ul>
      <li>Native PascalRAW gate: Perception fusion reduced FP and improved precision while recall drop stayed inside the current budget. This is a valid feasibility claim, not a universal superiority claim.</li>
      <li>RGB+Aux ablation: zeroing aux channels caused a large recall drop, so the compact detector is not simply ignoring aux. However, this is a compact fine-tuned detector, not yet a production detector.</li>
      <li>Matched YOLO seed2: RGB+Aux edge6 increased recall but did not improve mAP50-95 versus RGB-only, so it is evidence of a useful signal path rather than a broad detector win.</li>
      <li>Scene-edge suite: Perception RGB and aux strength align with high-information scene edge proxy, and PSF response behaves in the expected direction. The older aux confidence formulation is weaker, which is why edge evidence is now prioritized.</li>
    </ul>
    <div class="note warn">Recall/FP tradeoff 자체만으로는 강한 장점 주장이 어렵습니다. 강한 주장이 되려면 같은 recall budget에서 FP가 줄거나, 같은 FP budget에서 recall이 올라가거나, 특정 safety/task slice에서 유의미한 early-warning gain이 있어야 합니다.</div>
    <h3>Expectation level for unverified work</h3>
    <table>{_kv_rows(expectation_rows, headers=("Future validation", "Expected outcome", "Reasonable confidence", "Main risk"))}</table>
  </section>

  <section id="gaps" class="tab-panel">
    <h2>Unverified Claims and Reasons</h2>
    <table>{_gap_rows()}</table>
    <h3>Why some validation is still missing</h3>
    <p>가장 큰 이유는 데이터와 resource입니다. PascalRAW는 이미 좋은 native RAW evidence를 만들었지만 adverse/weather/low-light 다양성은 부족합니다. AODRaw는 benchmark로 강하지만 접근성과 용량 부담이 큽니다. LOD는 현실적으로 받을 수 있는 low-light RAW 후보지만, 먼저 파일이 native CR2인지 processed RAW-like인지 확인해야 합니다.</p>
  </section>

  <section id="roadmap" class="tab-panel">
    <h2>Next Work</h2>
    <table>{_roadmap_rows()}</table>
    <div class="note">지금 가장 빠르게 성능 개선 가능성을 확인하는 경로는 PascalRAW에서 edge6/RGB+Aux 학습을 정리하고, 동시에 LOD를 확보해 low-light hard slice로 검증하는 것입니다. 대규모 AODRaw 학습은 이후 단계가 맞습니다.</div>
  </section>

  <section id="sources" class="tab-panel">
    <h2>Source Evidence</h2>
    <p>이 HTML은 아래 local summary JSON을 읽어 생성했습니다.</p>
    <table>{source_links}</table>
    <h3>Claim boundaries copied from evidence reports</h3>
    <ul>{_boundary_list(_list(summary.get('claim_boundaries')))}</ul>
  </section>
</div>
<script>
  const buttons = Array.from(document.querySelectorAll('.tab-button'));
  const panels = Array.from(document.querySelectorAll('.tab-panel'));
  function activate(id) {{
    buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === id));
    panels.forEach(panel => panel.classList.toggle('active', panel.id === id));
    history.replaceState(null, '', '#' + id);
  }}
  buttons.forEach(btn => btn.addEventListener('click', () => activate(btn.dataset.tab)));
  const initial = location.hash ? location.hash.slice(1) : 'overview';
  if (document.getElementById(initial)) activate(initial);
</script>
</body>
</html>
"""


def _purpose_rows() -> list[tuple[str, str, str]]:
    return [
        (
            "Can ISP be optimized for machine perception rather than only human RGB?",
            "Build a HumanISP branch and a PerceptionISP branch from the same RAW/CFA source, then compare detector and edge behavior.",
            "Yes as a software research platform: both branches exist, provenance is tracked, and PascalRAW native held-out gates are runnable.",
        ),
        (
            "Does aux information have a practical role after the ISP?",
            "Export RGB+Aux tensors and train/evaluate DNN paths where aux is an input or proposal calibration signal.",
            "Yes for compact DNN input-dependence; not yet enough for production detector superiority.",
        ),
        (
            "Where should PerceptionISP help most?",
            "Focus on edge ambiguity, demosaic artifacts, low MTF/blur, low light, small/thin objects, and CFA/PSF-sensitive cases.",
            "Scene-edge and PSF diagnostics support this direction; stronger adverse RAW datasets are still needed.",
        ),
        (
            "What is the near-term engineering route?",
            "Use PascalRAW to fix training/evaluation correctness, then use LOD/AODRaw for low-light/adverse validation.",
            "PascalRAW is active; LOD is partial due Google Drive quota and needs archive completion before evaluation.",
        ),
    ]


def _claim_ladder_rows(
    native: Mapping[str, Any],
    dense: Mapping[str, Any],
    scene: Mapping[str, Any],
    yolo: Mapping[str, Any],
    lod: Mapping[str, Any],
    yolo_seed2: Mapping[str, Any],
) -> list[tuple[str, str, str, str]]:
    yolo_seed2_deltas = _mapping(yolo_seed2.get("deltas"))
    return [
        (
            "L1: Software feasibility",
            "Supported",
            "RAW/CFA ingestion, HumanISP, PerceptionISP, aux export, detector/fusion gates, and tabbed reports are implemented.",
            "Keep unit tests and provenance checks green as datasets expand.",
        ),
        (
            "L2: Native RAW FP/precision gain",
            "Supported for PascalRAW RGGB held-out",
            f"750 native RAW samples: dP50={_fmt(native.get('precision_delta_50'), signed=True)}, dFP50={_fmt(native.get('fp_delta_50'), signed=True)}, dR50={_fmt(native.get('recall_delta_50'), signed=True)}.",
            "Repeat across low-light/adverse RAW and hard object slices.",
        ),
        (
            "L3: DNN actually uses aux",
            "Supported for compact RGB+Aux model",
            f"Input ablation: zero-aux recall delta={_fmt(_mapping(_mapping(dense.get('deltas_vs_none')).get('zero_aux')).get('recall'), signed=True)} across {_int(dense.get('seed_count'))} seeds.",
            "Run matched YOLO/production-style fine-tuning with RGB-only baseline.",
        ),
        (
            "L4: Edge evidence helps under ambiguity",
            "Partially supported",
            f"Scene-edge aux evidence/strength improves edge F1 in proxy tests; Perception RGB dF1={_fmt(_mapping(scene.get('aggregate')).get('perception_rgb_minus_human_source_edge_f1_mean'), signed=True)}.",
            "Use object-boundary GT and low-MTF/blur/adverse slices rather than source-edge proxy only.",
        ),
        (
            "L5: Broad detector superiority",
            "Partially tested; not supported as broad superiority",
            (
                f"YOLO audit has {_int(yolo.get('fair_valid_edge6_rgb_comparison_count'))} fair comparisons. "
                f"Matched seed2 edge6 vs RGB: dR={_fmt(yolo_seed2_deltas.get('recall'), signed=True)}, "
                f"dAP50={_fmt(yolo_seed2_deltas.get('mAP50'), signed=True)}, "
                f"dAP50-95={_fmt(yolo_seed2_deltas.get('mAP50_95'), signed=True)}."
            ),
            "Hard-slice metrics, multi-seed stability, and production-scale detector gates are required.",
        ),
        (
            "L6: Adverse-condition RAW claim",
            "Pending",
            f"LOD status={lod.get('status', 'unknown')}; {lod.get('message', '')}",
            "Complete LOD or AODRaw acquisition, inspect native RAW format, then run low-light/adverse gates.",
        ),
    ]


def _expected_effect_rows(yolo_seed2: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    yolo_seed2_deltas = _mapping(yolo_seed2.get("deltas"))
    return [
        (
            "CFA/source provenance",
            "Tracks source CFA, target CFA, native mosaic flag, remap count, and source resolution.",
            "Prevents invalid claims caused by silently changing CFA pattern or using RGB proxy as native RAW.",
            "Strong in PascalRAW RGGB gate; all-CFA coverage still pending.",
        ),
        (
            "Edge-aware demosaic / RGB branch",
            "Produces perception-oriented RGB that preserves machine-useful edges and reduces false structures.",
            "Should reduce detector false positives from demosaic artifacts while preserving true boundaries.",
            "Supported by PascalRAW fusion FP reduction and scene-edge proxy.",
        ),
        (
            "Aux edge strength / edge evidence",
            "Adds explicit edge evidence channels beyond rendered RGB.",
            "Should help small/thin objects, weak boundaries, blur, and low-MTF cases where RGB texture is ambiguous.",
            (
                "Proxy supported. Matched YOLO seed2 shows recall gain "
                f"({_fmt(yolo_seed2_deltas.get('recall'), signed=True)}) but mAP50-95 loss "
                f"({_fmt(yolo_seed2_deltas.get('mAP50_95'), signed=True)}), so hard-slice targeting is still needed."
            ),
        ),
        (
            "Noise/artifact reliability",
            "Provides confidence-style channels for noisy or artifact-prone regions.",
            "Should let downstream logic suppress proposals caused by sensor/ISP artifacts.",
            "Partially supported by FP reduction; needs condition-specific adverse RAW validation.",
        ),
        (
            "PSF/blur-aware signals",
            "Encodes blur/PSF confidence and edge degradation behavior.",
            "Should expose when optical blur makes boundaries less reliable, especially under larger PSF.",
            "Diagnostic sweep behaves plausibly; stronger PSF conditions and native data are needed.",
        ),
        (
            "RGB+Aux DNN input path",
            "Changes the detector input from RGB-only to RGB plus selected aux channels while reusing downstream detector capacity where possible.",
            "Should allow the network to learn when aux evidence corrects or calibrates RGB proposals.",
            "Compact DNN dependence supported; matched YOLO seed2 reached near parity with recall gain but not mAP50-95 gain.",
        ),
    ]


def _dataset_test_rows(
    native: Mapping[str, Any],
    dense: Mapping[str, Any],
    scene: Mapping[str, Any],
    yolo: Mapping[str, Any],
    lod: Mapping[str, Any],
    raw: Mapping[str, Any],
    lod_ready: Mapping[str, Any],
    yolo_seed2: Mapping[str, Any],
) -> list[tuple[str, str, str, str]]:
    yolo_seed2_deltas = _mapping(yolo_seed2.get("deltas"))
    return [
        (
            "PascalRAW native held-out 750",
            "Main HumanISP vs PerceptionISP native RAW claim gate.",
            f"Pass: dP50={_fmt(native.get('precision_delta_50'), signed=True)}, dFP50={_fmt(native.get('fp_delta_50'), signed=True)}, dR50={_fmt(native.get('recall_delta_50'), signed=True)}.",
            "Mostly RGGB and not a broad adverse-condition dataset.",
        ),
        (
            "PascalRAW native1581 compact DNN",
            "Check whether RGB+Aux input is actually used by a learned model.",
            f"Pass across {_int(dense.get('seed_count'))} seeds; zero-aux recall delta={_fmt(_mapping(_mapping(dense.get('deltas_vs_none')).get('zero_aux')).get('recall'), signed=True)}.",
            "Compact detector, low absolute precision, and not production YOLO-scale proof.",
        ),
        (
            "Scene-edge CFA/PSF sweep",
            "Measure whether edge information from HumanISP/PerceptionISP follows source scene edges.",
            f"{_int(scene.get('sample_count')) or 12} cases; aux strength and perception RGB beat HumanISP edge proxy on average.",
            "Uses source-edge proxy, not object-boundary ground truth.",
        ),
        (
            "YOLO edge6 audit",
            "Verify whether YOLO edge6 results really used aux_edge_evidence and fair training recipes.",
            f"{_int(yolo.get('fair_valid_edge6_rgb_comparison_count'))} fair rows; {_int(yolo.get('training_mismatched_valid_edge6_rgb_comparison_count'))} mismatched rows flagged.",
            "Current strongest YOLO gains are not yet fair performance claims.",
        ),
        (
            "Matched YOLO seed2 RGB vs edge6",
            "Check a fair full PascalRAW native1581 YOLO11n comparison with same seed, epochs, optimizer, image size, and split.",
            (
                f"edge6 dR={_fmt(yolo_seed2_deltas.get('recall'), signed=True)}, "
                f"dAP50={_fmt(yolo_seed2_deltas.get('mAP50'), signed=True)}, "
                f"dAP50-95={_fmt(yolo_seed2_deltas.get('mAP50_95'), signed=True)}."
            ),
            "Positive recall signal, but whole-val localization/AP quality is not better than RGB-only.",
        ),
        (
            "LOD low-light RAW",
            "Practical next dataset for low-light/adverse validation.",
            f"{lod.get('status', 'unknown')}: {lod.get('message', '')}; readiness={lod_ready.get('status', 'unknown')}.",
            "Download is incomplete and Google Drive quota blocked the file.",
        ),
        (
            "AODRaw",
            "Large adverse-condition object detection benchmark candidate.",
            _aodraw_note(raw),
            "Large storage/network burden; current local RAW image zip is missing.",
        ),
    ]


def _expectation_rows(yolo_seed2: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    yolo_seed2_deltas = _mapping(yolo_seed2.get("deltas"))
    return [
        (
            "LOD low-light subset",
            "PerceptionISP should show clearer benefit than generic PascalRAW if low-light edges and noise artifacts are common.",
            "Medium",
            "LOD may be processed RAW-like rather than native CR2, and annotation/image alignment must be verified.",
        ),
        (
            "Hard small/thin object slice",
            "Aux edge evidence should help more on small, thin, long, weak-boundary objects than on large easy objects.",
            "Medium-high",
            "The available annotations may not contain enough such objects, so mining quality matters.",
        ),
        (
            "Matched YOLO edge6 multi-seed",
            "If aux evidence is useful as model input, matched edge6 should beat RGB-only in at least hard-slice metrics, even if whole-val gain is small.",
            "Medium-low for whole-val; medium for hard-slice",
            (
                f"Seed2 gave dR={_fmt(yolo_seed2_deltas.get('recall'), signed=True)} but "
                f"dAP50-95={_fmt(yolo_seed2_deltas.get('mAP50_95'), signed=True)}. "
                "RGB-only may already be strong enough on easy PascalRAW; aux likely needs hard-case mining or better pretraining."
            ),
        ),
        (
            "CFA-specific utility",
            "CFA patterns with weaker sampling for relevant colors/edges should expose stronger PerceptionISP gains.",
            "Medium",
            "Synthetic remosaic can exaggerate or distort native sensor behavior.",
        ),
        (
            "PSF-aware evidence",
            "Larger meaningful PSF blur should make edge ambiguity visible and make aux evidence more valuable.",
            "Medium",
            "If PSF kernels are too small or unrealistic, the effect will remain weak.",
        ),
        (
            "Production detector fine-tune",
            "RGB+Aux should improve calibration or hard-slice recall/FP if trained with enough matched native/adverse RAW.",
            "Unknown-medium",
            "Resource and data volume may dominate; a small laptop run may not be enough.",
        ),
    ]


def _implemented_rows() -> str:
    rows = [
        ("RAW/CFA ingestion", "PascalRAW native NEF loader, CameraE2E bridge, CFA pattern provenance, source/target CFA match checks."),
        ("HumanISP baseline", "RGB baseline branch for detector comparison and RGB edge proxy comparison."),
        ("PerceptionISP RGB", "Edge-aware demosaic, denoise, tone mapping, artifact suppression, perception-oriented RGB output."),
        ("Aux map generation", "Edge strength, edge confidence, edge evidence, PSF/blur likelihood, artifact/noise reliability maps, extended tensor export."),
        ("DNN export", "RGB+Aux HWC/CHW NPZ export, custom channel lists, edge6 export, YOLO dataset conversion."),
        ("Detector/fusion gates", "HumanISP vs PerceptionISP comparison, proposal calibration, claim gates, native held-out audit."),
        ("Training path", "Compact RGB+Aux detector fine-tuning, RGB initialization / zero-aux initialization experiments, input ablation."),
        ("Dataset acquisition", "PascalRAW in use, AODRaw/SID/LOD acquisition/readiness tools, LOD download path in progress."),
    ]
    return _kv_rows(rows, headers=("Block", "Implemented behavior"))


def _gap_rows() -> str:
    rows = [
        ("All-CFA claim", "Not verified", "Current strong native benchmark is RGGB-heavy. Need RGGB/GRBG/BGGR/GBRG controlled sweep with native or credible remosaic data."),
        ("Adverse-condition broad claim", "Not verified", "PascalRAW lacks enough night/rain/fog/glare/HDR diversity. LOD/AODRaw must be acquired and audited."),
        ("Production detector superiority", "Not verified", "Compact RGB+Aux DNN shows aux dependence, but a production-scale detector fine-tune and held-out gate are still needed."),
        ("Aux confidence as final signal", "Partially contradicted", "Scene-edge evidence shows aux strength/evidence are useful, while old aux confidence underperforms in some summaries."),
        ("YOLO edge6 naming", "Partially corrected", "Audit found some edge6-named w640 runs that did not contain aux_edge_evidence. Only channel_status=edge6_valid rows should be cited."),
        ("Native LOD claim", "Pending", "LOD package is downloading. Must inspect whether files are native Canon CR2 or processed RAW-like images."),
        ("Small/thin object gain", "Not proven", "Existing PascalRAW slices have weak small-object signal. Need hard-object mining or LOD/AODRaw task-specific slices."),
    ]
    return _kv_rows(rows, headers=("Claim", "Status", "Reason"))


def _roadmap_rows() -> str:
    rows = [
        ("P0", "Finish LOD download and inspect archive", "Check ZIP integrity, list directories, identify CR2/native RAW vs processed RAW, update readiness."),
        ("P0", "Run LOD low-light smoke subset", "Use 32-128 images first: decode RAW, compare HumanISP/PerceptionISP, verify annotation coordinate alignment."),
        ("P1", "Re-export PascalRAW edge6 tensors", "Use edge_evidence in base RGB+Aux tensor, then rerun compact training/eval with matched RGB-only baseline."),
        ("P1", "Hard-slice mining", "Create small/thin/low-MTF/demosaic-artifact slices where edge aux should matter most."),
        ("P1", "CFA/LensPSF sweep", "Separate CFA-specific benefit and show stronger effect under meaningful PSF blur."),
        ("P2", "AODRaw adverse benchmark", "After disk/access are stable, use AODRaw for broad adverse-condition claim gate."),
        ("P2", "Production detector fine-tune", "Train a stronger RGB+Aux detector with adequate data, then run held-out mAP/APs/APm/APl gates."),
    ]
    return _kv_rows(rows, headers=("Priority", "Action", "Evidence it should produce"))


def _dense_rows(dense: Mapping[str, Any]) -> list[tuple[str, str]]:
    means = _mapping(dense.get("mean_by_mode"))
    deltas = _mapping(dense.get("deltas_vs_none"))
    none = _mapping(means.get("none"))
    zero_aux = _mapping(deltas.get("zero_aux"))
    zero_rgb = _mapping(deltas.get("zero_rgb"))
    shuffle = _mapping(deltas.get("shuffle_aux"))
    return [
        ("Dataset/split", f"PascalRAW native 1581 union, held-out test samples={_int(dense.get('test_sample_count'))}, seeds={_int(dense.get('seed_count'))}."),
        ("Full RGB+Aux mean", f"precision={_fmt(none.get('precision'))}, recall={_fmt(none.get('recall'))}, FP/image={_fmt(none.get('fp'))}."),
        ("Zero aux", f"recall delta={_fmt(zero_aux.get('recall'), signed=True)}, precision delta={_fmt(zero_aux.get('precision'), signed=True)}."),
        ("Zero RGB", f"recall delta={_fmt(zero_rgb.get('recall'), signed=True)}, precision delta={_fmt(zero_rgb.get('precision'), signed=True)}."),
        ("Shuffle aux", f"recall delta={_fmt(shuffle.get('recall'), signed=True)}. This is diagnostic and not a standalone performance claim."),
        ("Claim boundary", "Shows aux input dependence for compact DNN; does not prove production detector superiority."),
    ]


def _scene_rows(scene: Mapping[str, Any]) -> list[tuple[str, str]]:
    agg = _mapping(scene.get("aggregate"))
    best = _mapping(scene.get("aux_best_candidate"))
    return [
        ("Scene-edge proxy", "High-information source scene edge map used as proxy oracle; not object-boundary GT."),
        ("Perception RGB vs Human RGB", f"dF1={_fmt(agg.get('perception_rgb_minus_human_source_edge_f1_mean'), signed=True)}, win-rate={_fmt(agg.get('perception_rgb_source_edge_f1_win_rate'))}."),
        ("Aux strength vs Human RGB", f"dF1={_fmt(agg.get('perception_aux_strength_minus_human_source_edge_f1_mean'), signed=True)}, win-rate={_fmt(agg.get('perception_aux_strength_source_edge_f1_win_rate'))}."),
        ("Old aux confidence weakness", f"dF1={_fmt(agg.get('perception_aux_confidence_minus_human_source_edge_f1_mean'), signed=True)}; this is why edge_evidence became the preferred channel."),
        ("Edge-evidence diagnostic", f"best={html_lib.escape(str(best.get('name', 'unknown')))}, dF1={_fmt(best.get('minus_human_source_edge_f1_mean'), signed=True)}, negative separation count={_int(best.get('negative_scene_edge_separation_count'))}."),
        ("LensPSF response", "Higher PSF sigma reduced edge confidence in the expected direction in the CFA/PSF sweep."),
    ]


def _matched_yolo_seed2(root: Path) -> dict[str, Any]:
    rgb_path = root / YOLO_MATCHED_RUNS["rgb_only_seed2"]
    edge_path = root / YOLO_MATCHED_RUNS["rgb_aux_edge6_seed2"]
    rgb = _read_json(rgb_path)
    edge = _read_json(edge_path)
    if rgb.get("status") == "missing" or edge.get("status") == "missing":
        return {
            "status": "missing",
            "rgb_summary": str(rgb_path),
            "edge6_summary": str(edge_path),
            "message": "Matched RGB-only and RGB+Aux edge6 seed2 summaries are not both available.",
        }

    rgb_metrics = _yolo_metrics(rgb)
    edge_metrics = _yolo_metrics(edge)
    deltas = {name: edge_metrics.get(name, 0.0) - rgb_metrics.get(name, 0.0) for name in rgb_metrics}
    return {
        "status": "partial",
        "rgb_summary": str(rgb_path),
        "edge6_summary": str(edge_path),
        "comparison": "YOLO11n PascalRAW native1581 full, seed2, 20 epochs, imgsz512, batch8, AdamW lr0=5e-4, MPS.",
        "rgb_only": rgb_metrics,
        "rgb_aux_edge6": edge_metrics,
        "deltas": deltas,
        "interpretation": (
            "RGB+Aux edge6 improves recall and almost ties mAP50, but loses mAP50-95 versus RGB-only. "
            "This supports the aux-input path and a possible recall tradeoff, not broad detector superiority."
        ),
        "aux_adapter": _mapping(edge.get("aux_feature_adapter_init")),
    }


def _yolo_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    results = _mapping(summary.get("results_dict"))
    return {
        "precision": float(results.get("metrics/precision(B)", 0.0)),
        "recall": float(results.get("metrics/recall(B)", 0.0)),
        "mAP50": float(results.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results.get("metrics/mAP50-95(B)", 0.0)),
    }


def _yolo_seed2_rows(yolo_seed2: Mapping[str, Any]) -> list[tuple[str, str]]:
    rgb = _mapping(yolo_seed2.get("rgb_only"))
    edge = _mapping(yolo_seed2.get("rgb_aux_edge6"))
    deltas = _mapping(yolo_seed2.get("deltas"))
    adapter = _mapping(yolo_seed2.get("aux_adapter"))
    return [
        ("Protocol", str(yolo_seed2.get("comparison", "Matched seed2 summaries not available."))),
        ("RGB-only", f"P={_fmt(rgb.get('precision'))}, R={_fmt(rgb.get('recall'))}, mAP50={_fmt(rgb.get('mAP50'))}, mAP50-95={_fmt(rgb.get('mAP50_95'))}."),
        ("RGB+Aux edge6", f"P={_fmt(edge.get('precision'))}, R={_fmt(edge.get('recall'))}, mAP50={_fmt(edge.get('mAP50'))}, mAP50-95={_fmt(edge.get('mAP50_95'))}."),
        ("Delta edge6 - RGB", f"dP={_fmt(deltas.get('precision'), signed=True)}, dR={_fmt(deltas.get('recall'), signed=True)}, dAP50={_fmt(deltas.get('mAP50'), signed=True)}, dAP50-95={_fmt(deltas.get('mAP50_95'), signed=True)}."),
        ("Aux initialization", f"feature_adapter={adapter.get('status', 'unknown')}, scale={adapter.get('scale', 'n/a')}, aux_channels={adapter.get('aux_channel_count', 'n/a')}."),
        ("Interpretation", str(yolo_seed2.get("interpretation", ""))),
    ]


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
    return {"status": "missing", "path": "", "size_bytes": 0, "expected_bytes": LOD_EXPECTED_BYTES, "progress": 0.0, "message": "LOD_BMVC2021 archive not found locally."}


def _overall_status(*summaries: Mapping[str, Any]) -> str:
    if all(str(summary.get("status", "")).lower() == "pass" for summary in summaries):
        return "pass"
    if any(str(summary.get("status", "")).lower() in {"fail", "blocked"} for summary in summaries):
        return "partial"
    return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "missing_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _card(row: Mapping[str, Any]) -> str:
    status = html_lib.escape(str(row.get("status", "unknown")))
    return (
        f"<article class=\"card\"><div class=\"label\">{html_lib.escape(str(row.get('label', '')))}</div>"
        f"<div class=\"value\">{html_lib.escape(str(row.get('value', '')))}</div>"
        f"<div class=\"detail\">{html_lib.escape(str(row.get('detail', '')))}</div>"
        f"<p><span class=\"status {status}\">{status}</span></p></article>"
    )


def _tab_button(tab_id: str, label: str, active: bool) -> str:
    cls = "tab-button active" if active else "tab-button"
    return f"<button class=\"{cls}\" type=\"button\" data-tab=\"{html_lib.escape(tab_id)}\">{html_lib.escape(label)}</button>"


def _kv_rows(rows: Sequence[Sequence[Any]], headers: Sequence[str] = ("Item", "Evidence")) -> str:
    head = "".join(f"<th>{html_lib.escape(str(item))}</th>" for item in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html_lib.escape(str(item))}</td>" for item in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"


def _source_links(paths: Mapping[str, Any]) -> str:
    rows = []
    for name, path in paths.items():
        html_path = str(path).replace("_summary.json", "").replace(".json", "")
        if "claim_dashboard_summary" in str(path):
            html_path = "reports/perception_claim_dashboard_pascalraw_native1581_rgb_aux_3seed_learned_evidence_v1/index.html"
        elif "native_heldout_benchmark_audit_summary" in str(path):
            html_path = "reports/perception_native_heldout_benchmark_audit_pascalraw_rawdet7_val750_fusion_v2/index.html"
        elif "dense_input_ablation_summary" in str(path):
            html_path = "reports/perception_dense_input_ablation_gate_pascalraw_native1581_union_seed101_202_303_broad_cached_v1/index.html"
        elif "scene_edge_confidence_summary" in str(path):
            html_path = "reports/perception_scene_edge_confidence_bus_cfa_psf_sweep/index.html"
        elif "scene_edge_aux_sweep_summary" in str(path):
            html_path = "reports/perception_scene_edge_aux_sweep_sid_lowlight_native16_edge_evidence_v1/index.html"
        elif "yolo_evidence_audit_summary" in str(path):
            html_path = "reports/perception_yolo_edge6_evidence_audit_current_v1/index.html"
        elif "raw_dataset_acquisition_summary" in str(path):
            html_path = "reports/perception_raw_dataset_acquisition_lod_priority_v1/index.html"
        elif "lod_local_readiness_summary" in str(path):
            html_path = "reports/perception_lod_local_readiness_current_v1/index.html"
        rows.append((name, path, html_path))
    return _kv_rows(rows, headers=("Evidence", "Summary JSON", "HTML report"))


def _claim_list(claims: Sequence[Any]) -> str:
    items = []
    for claim in claims:
        if isinstance(claim, Mapping):
            text = f"{claim.get('status', 'unknown')}: {claim.get('claim', '')}"
        else:
            text = str(claim)
        items.append(f"<li>{html_lib.escape(text)}</li>")
    return "".join(items)


def _boundary_list(boundaries: Sequence[Any]) -> str:
    return "".join(f"<li>{html_lib.escape(str(item))}</li>" for item in boundaries if str(item).strip())


def _aodraw_note(raw: Mapping[str, Any]) -> str:
    local = _mapping(raw.get("local_state"))
    annotations = bool(local.get("aodraw_annotations_present"))
    test_raw = _mapping(local.get("aodraw_test_raw_zip")).get("status", "unknown")
    srgb = _mapping(local.get("aodraw_srgb_zip")).get("status", "unknown")
    return f"annotations_present={annotations}, test_raw={test_raw}, srgb={srgb}; raw test zip still needed for evaluation."


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _fmt(value: Any, *, signed: bool = False) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number:.4f}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
