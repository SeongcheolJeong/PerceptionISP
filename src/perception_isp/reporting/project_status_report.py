"""Generate a current, decision-oriented PerceptionISP status report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from perception_isp.core.aux_map_catalog import AUX_MAP_SPECS
from perception_isp.core.aux_map_rationale import (
    AUX_MAP_RATIONALE_BY_NAME,
    AUX_MAP_RATIONALES,
)
from perception_isp.core.paths import camerae2e_source
from perception_isp.reporting.project_accomplishment_report import (
    build_report as build_accomplishment_report,
)


SCHEMA_VERSION = "perception_isp_project_status_v1"
SHOWCASE_SUMMARY = Path(
    "reports/perception_nuscenes_scene0061_camerae2e_pipeline_showcase_v1/summary.json"
)
DEFAULT_OUTPUT = Path("reports/perception_project_status_current_v1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the detailed PerceptionISP purpose, status, gap, and roadmap report."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--verified-tests", type=int)
    parser.add_argument("--verified-skipped", type=int)
    parser.add_argument("--verified-subtests", type=int)
    parser.add_argument(
        "--camerae2e-smoke",
        choices=("pass", "fail", "not-run"),
        default="not-run",
    )
    args = parser.parse_args(argv)
    root = Path(args.project_root).expanduser().resolve(strict=True)
    report = build_status_report(
        root,
        verified_tests=args.verified_tests,
        verified_skipped=args.verified_skipped,
        verified_subtests=args.verified_subtests,
        camerae2e_smoke=str(args.camerae2e_smoke),
    )
    path = write_status_report(report, Path(args.output_dir).expanduser())
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(path.resolve()),
                "summary_json": str(path.with_name("project_status_summary.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_status_report(
    root: Path,
    *,
    verified_tests: int | None = None,
    verified_skipped: int | None = None,
    verified_subtests: int | None = None,
    camerae2e_smoke: str = "not-run",
) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    accomplishment = build_accomplishment_report(root)
    showcase_path = root / SHOWCASE_SUMMARY
    showcase = _read_json(showcase_path)
    camera_root = camerae2e_source()
    camera_repo = (
        camera_root.parent
        if camera_root.name == "src" and camera_root.parent.exists()
        else camera_root
    )
    aux_groups: dict[str, int] = {}
    for spec in AUX_MAP_SPECS:
        aux_groups[spec.group] = aux_groups.get(spec.group, 0) + 1
    showcase_metrics = showcase.get("metrics", {}) if isinstance(showcase, Mapping) else {}
    native = accomplishment.get("native", {})
    dense = accomplishment.get("dense", {})
    scene = accomplishment.get("scene", {})
    yolo = accomplishment.get("yolo_seed2", {})
    hard = accomplishment.get("hard_fixedopt_gate1_e2_seed3", {})
    validation_status = (
        "pass"
        if verified_tests is not None and int(verified_tests) > 0 and camerae2e_smoke == "pass"
        else "partial"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "title": "PerceptionISP 목적·현재 진행상황·다음 단계",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial",
        "project_root": str(root),
        "executive_conclusion": (
            "PerceptionISP는 sensor-aware software reference와 evidence pipeline으로서는 완성도가 높은 단계다. "
            "다만 production ISP 또는 HumanISP 대비 일반 성능 우월성을 주장할 단계는 아니며, 다음 핵심은 "
            "native multi-exposure RAW·실센서 calibration·공정한 multi-seed task benchmark·temporal registration이다."
        ),
        "validation": {
            "status": validation_status,
            "pytest_passed": verified_tests,
            "pytest_skipped": verified_skipped,
            "pytest_subtests_passed": verified_subtests,
            "camerae2e_python312_runtime_smoke": camerae2e_smoke,
            "showcase_status": showcase.get("status", "missing"),
        },
        "showcase": {
            "path": str(SHOWCASE_SUMMARY),
            "status": showcase.get("status", "missing"),
            "source_frames": showcase_metrics.get("source_frame_count"),
            "pipeline_results": showcase_metrics.get("result_count"),
            "aux_maps_per_result": showcase_metrics.get("aux_map_count_per_result"),
            "aux_map_artifacts": showcase_metrics.get("aux_map_artifact_count"),
            "cache_hits": showcase_metrics.get("cache_hits"),
            "native_hdr_quality_claim_eligible": showcase.get(
                "native_hdr_quality_claim_eligible", False
            ),
        },
        "software": {
            "aux_map_count": len(AUX_MAP_SPECS),
            "aux_rationale_count": len(AUX_MAP_RATIONALES),
            "aux_groups": aux_groups,
            "core_python": "3.11+",
            "camerae2e_python": "3.12+",
            "perceptionisp_git": _git_snapshot(root),
            "camerae2e_git": _git_snapshot(camera_repo),
        },
        "evidence": {
            "overall_status": accomplishment.get("status", "missing"),
            "native_raw": native,
            "dense_aux_ablation": dense,
            "scene_edge": scene,
            "yolo_seed2": yolo,
            "hard_slice_seed3": hard,
        },
        "milestones": _milestone_rows(showcase, accomplishment, camerae2e_smoke),
        "roadmap": _roadmap_rows(),
        "sources": {
            "architecture": "docs/ARCHITECTURE.md",
            "evidence_boundaries": "docs/EVIDENCE_AND_LIMITATIONS.md",
            "user_guide": "docs/USER_GUIDE_KO.md",
            "aux_catalog": "src/perception_isp/core/aux_map_catalog.py",
            "aux_rationale": "src/perception_isp/core/aux_map_rationale.py",
            "pipeline": "src/perception_isp/core/pipeline.py",
            "showcase_report": (
                "reports/perception_nuscenes_scene0061_camerae2e_pipeline_showcase_v1/index.html"
            ),
            "historical_accomplishment_report": (
                "reports/perception_project_accomplishment_tabs_current_v1/index.html"
            ),
        },
    }
    return report


def write_status_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = destination / "project_status_summary.json"
    index = destination / "index.html"
    summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    index.write_text(_render_html(report), encoding="utf-8")
    return index


def _render_html(report: Mapping[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    validation = _mapping(report.get("validation"))
    showcase = _mapping(report.get("showcase"))
    software = _mapping(report.get("software"))
    evidence = _mapping(report.get("evidence"))
    native = _mapping(evidence.get("native_raw"))
    dense = _mapping(evidence.get("dense_aux_ablation"))
    scene = _mapping(evidence.get("scene_edge"))
    yolo = _mapping(evidence.get("yolo_seed2"))
    hard = _mapping(evidence.get("hard_slice_seed3"))
    hard_all = _mapping(_mapping(hard.get("filters")).get("all"))
    hard_small = _mapping(_mapping(hard.get("filters")).get("small"))
    hard_thin = _mapping(_mapping(hard.get("filters")).get("thin"))
    p_git = _mapping(software.get("perceptionisp_git"))
    c_git = _mapping(software.get("camerae2e_git"))
    milestone_rows = "".join(
        _table_row(
            row.get("area"),
            _badge(row.get("status"), esc),
            row.get("completed"),
            row.get("remaining"),
            esc=esc,
            raw_columns={1},
        )
        for row in report.get("milestones", [])
    )
    roadmap_rows = "".join(
        _table_row(
            row.get("priority"),
            row.get("work"),
            row.get("why"),
            row.get("acceptance"),
            row.get("depends_on"),
            esc=esc,
        )
        for row in report.get("roadmap", [])
    )
    test_value = "미기록"
    if validation.get("pytest_passed") is not None:
        skipped = validation.get("pytest_skipped")
        skipped_text = "" if skipped is None else f" · {skipped} skipped"
        test_value = (
            f"{validation.get('pytest_passed')} passed{skipped_text} · "
            f"{validation.get('pytest_subtests_passed', 0)} subtests"
        )
    architecture_svg = _architecture_svg()
    aux_rationale_html = _render_aux_rationales(esc)
    return f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>{esc(report.get('title'))}</title><style>
    :root{{--ink:#18242c;--muted:#64737d;--paper:#f3f6f4;--card:#fff;--line:#dce4e0;--accent:#0d6a67;--accent2:#1d4f91;--good:#18794e;--warn:#986008;--bad:#b33a2b}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 system-ui,-apple-system,sans-serif}}main{{max-width:1440px;margin:auto;padding:26px 34px 70px}}h1{{font-size:clamp(34px,5vw,64px);line-height:1.04;margin:.18em 0}}h2{{font-size:27px;margin:0 0 16px}}h3{{margin:28px 0 10px}}p{{max-width:1100px}}code{{background:#edf1ef;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}}a{{color:var(--accent2)}}
    .hero,.panel{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px}}.eyebrow{{color:var(--accent);font-weight:850;letter-spacing:.04em}}.lead{{font-size:18px;color:#33434d;max-width:1200px}}.conclusion{{border-left:6px solid var(--accent);background:#eaf4f1;padding:15px 18px;border-radius:8px}}
    .tabs{{position:sticky;top:0;z-index:10;display:flex;gap:8px;overflow-x:auto;margin:20px -3px 30px;padding:10px 3px;background:var(--paper)}}.tab{{white-space:nowrap;border:1px solid #c8d4cf;background:#fff;padding:10px 16px;border-radius:999px;font-weight:780;cursor:pointer}}.tab[aria-selected='true']{{background:var(--accent);border-color:var(--accent);color:white}}.tabpanel[hidden]{{display:none}}
    .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:11px;margin:14px 0 24px}}.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px}}.card b{{display:block;font-size:25px;color:var(--accent)}}.card span{{color:var(--muted)}}
    .flow{{display:grid;grid-template-columns:repeat(6,1fr);gap:9px}}.node{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;min-height:130px}}.node strong{{display:block;color:var(--accent);margin-bottom:6px}}.arrow{{display:none}}
    .diagram{{overflow-x:auto;background:linear-gradient(145deg,#f9fcfb,#eef5f3);border:1px solid var(--line);border-radius:16px;padding:12px;margin:14px 0 24px}}.diagram svg{{display:block;width:100%;min-width:1040px;height:auto}}.diagram text{{font-family:system-ui,-apple-system,sans-serif}}.diagram .box{{fill:#fff;stroke:#a9beb6;stroke-width:1.5}}.diagram .source{{fill:#eef4fb;stroke:#8daed3}}.diagram .core{{fill:#e8f4f0;stroke:#67a79a}}.diagram .output{{fill:#fff7e5;stroke:#d7ad56}}.diagram .gate{{fill:#f3edfb;stroke:#a988cf}}.diagram .title{{font-size:15px;font-weight:800;fill:#18313a}}.diagram .sub{{font-size:11px;fill:#52656f}}.diagram .wire{{fill:none;stroke:#58766d;stroke-width:2;marker-end:url(#arrowhead)}}.diagram .sidewire{{fill:none;stroke:#8b6ab3;stroke-width:1.8;stroke-dasharray:5 4;marker-end:url(#arrowhead-purple)}}
    .architecture-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:14px 0 24px}}.stage{{background:#fff;border:1px solid var(--line);border-top:4px solid var(--accent);border-radius:12px;padding:14px;min-height:145px}}.stage b{{display:block;color:var(--accent);margin-bottom:5px}}.formula-strip{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0 24px}}.formula-box{{background:#17262e;color:#e8f5f0;border-radius:12px;padding:15px}}.formula-box b{{color:#8cd7c5}}.formula-box code{{display:block;background:transparent;color:#fff;padding:7px 0;white-space:normal}}
    .aux-principles{{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;margin:14px 0 24px}}.principle{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:15px}}.principle b{{display:block;color:var(--accent2)}}.aux-group{{background:#fff;border:1px solid var(--line);border-radius:14px;margin:13px 0;overflow:hidden}}.aux-group>summary{{cursor:pointer;list-style:none;padding:17px 19px;background:#eaf1ee;font-size:18px;font-weight:850}}.aux-group>summary::-webkit-details-marker{{display:none}}.aux-group>summary::after{{content:'＋';float:right;color:var(--accent)}}.aux-group[open]>summary::after{{content:'−'}}.aux-group-intro{{padding:15px 18px 0;color:#455963}}.aux-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;padding:16px}}.aux-card{{border:1px solid var(--line);border-radius:12px;padding:16px;background:#fbfdfc}}.aux-card h3{{display:flex;align-items:center;gap:8px;margin:0 0 12px;font-size:18px}}.aux-card h4{{margin:13px 0 4px;font-size:13px;color:var(--accent);letter-spacing:.02em}}.aux-card p{{margin:0}}.aux-card pre{{margin:7px 0 10px;padding:12px;background:#17262e;color:#f1f7f5;border-radius:9px;white-space:pre-wrap;overflow-wrap:anywhere;font:12.5px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}}.tag{{display:inline-block;border:1px solid #c7d8d1;border-radius:99px;padding:2px 8px;color:#51655d;background:#f5f9f7;font-size:11px;font-weight:750}}.boundary{{border-left:4px solid var(--warn);background:#fff8e8;padding:10px 12px;margin-top:10px!important}}.audit-list{{columns:2;column-gap:32px}}.audit-list li{{break-inside:avoid;margin:.55em 0}}
    .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.callout{{border-left:5px solid var(--accent2);background:#eef4fb;padding:13px 16px;margin:14px 0}}.warning{{border-left-color:var(--warn);background:#fff7e5}}.danger{{border-left-color:var(--bad);background:#fff0ed}}
    .table-wrap{{overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:12px;margin:12px 0 24px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:11px 13px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}}th{{background:#eaf1ee}}.badge{{display:inline-block;border-radius:99px;padding:2px 9px;font-size:12px;font-weight:850;text-transform:uppercase;background:#edf1ef}}.badge.pass,.badge.complete,.badge.implemented{{color:var(--good);background:#e4f4eb}}.badge.partial,.badge.stress-only{{color:var(--warn);background:#fff1cf}}.badge.open,.badge.missing,.badge.fail{{color:var(--bad);background:#fde5df}}
    .checklist li{{margin:.55em 0}}.links{{display:flex;flex-wrap:wrap;gap:9px}}.links a{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:9px 12px;text-decoration:none}}.small{{font-size:13px;color:var(--muted)}}
    @media(max-width:900px){{main{{padding:18px}}.grid2,.flow,.architecture-grid,.formula-strip,.aux-principles,.aux-grid{{grid-template-columns:1fr}}.audit-list{{columns:1}}.tabs{{margin-left:-18px;margin-right:-18px;padding-left:18px;padding-right:18px}}}}</style></head><body><main>
    <header class='hero'><div class='eyebrow'>CURRENT PROJECT STATUS · {esc(str(report.get('generated_at'))[:10])}</div><h1>{esc(report.get('title'))}</h1><p class='lead'>PerceptionISP가 왜 필요한지, 지금 어디까지 구현·검증됐는지, 다음 의사결정 전에 무엇을 완료해야 하는지를 하나의 보고서로 정리했습니다.</p><p class='conclusion'><b>현재 결론:</b> {esc(report.get('executive_conclusion'))}</p></header>
    <nav class='tabs' role='tablist' aria-label='프로젝트 상태 보고서'>
      {_tab('purpose','1. 목적·판단기준',True)}{_tab('architecture','2. 아키텍처',False)}{_tab('aux','3. 33 Aux 수식',False)}{_tab('status','4. 현재 진행상황',False)}{_tab('evidence','5. 검증·성과',False)}{_tab('limits','6. 한계·위험',False)}{_tab('next','7. 다음 해야 할 일',False)}{_tab('use','8. 실행·자료',False)}
    </nav>
    <section class='tabpanel' id='purpose' role='tabpanel' aria-labelledby='tab-purpose'>
      <h2>1. PerceptionISP의 목적</h2><div class='grid2'><div class='panel'><h3>문제 정의</h3><p>전통적인 ISP는 사람이 보기 자연스러운 RGB를 주 목표로 삼습니다. 하지만 detector·segmentation·SLAM 같은 machine perception은 예쁜 영상보다 경계 보존, 포화·노이즈·demosaic 불확실성, HDR source 선택, 시간 안정성 같은 정보를 필요로 합니다.</p><p>PerceptionISP는 같은 RAW에서 <b>Human-view RGB</b>와 <b>machine-view RGB + Aux evidence</b>를 동시에 만들어, 어느 정보가 downstream task에 도움이 되는지 공정하게 연구하는 software reference입니다.</p></div><div class='panel'><h3>성공의 정의</h3><ul class='checklist'><li>하나의 sensor-domain source of truth와 감사 가능한 provenance</li><li>동일 scene·RAW·annotation·모델·운영점의 matched comparison</li><li>성능 gain과 FP·latency·calibration trade-off 동시 보고</li><li>유리한 결과뿐 아니라 counterexample과 claim boundary 유지</li><li>native RAW와 pseudo-RAW를 절대 혼용하지 않음</li></ul></div></div>
      <h3>범위</h3><div class='table-wrap'><table><thead><tr><th>포함</th><th>포함하지 않음</th></tr></thead><tbody><tr><td>RAW normalization, HDR, CFA/demosaic, color, noise, edge/optics, temporal evidence, Human/Machine RGB, Aux tensor, detection/segmentation evaluation</td><td>고수준 행동계획, 범용 detector 대체, 생산차량 ISP 품질 보장, native RAW 복원용 inverse ISP, 자동으로 보장되는 task 성능 향상</td></tr></tbody></table></div>
      <div class='callout warning'><b>핵심 판단:</b> “33개 map을 만들 수 있다”는 소프트웨어 실행 가능성이고, “33개 map이 실제 perception을 개선한다”는 별도의 held-out task 가설입니다.</div>
    </section>
    <section class='tabpanel' id='architecture' role='tabpanel' aria-labelledby='tab-architecture' hidden>
      <h2>2. 실제 코드 기준 아키텍처</h2><p class='lead'>핵심 설계는 “RGB를 더 예쁘게 만드는 ISP”가 아니라, RGB를 만들며 사라지는 <b>sensor evidence와 불확실성</b>을 공간 정렬된 map으로 보존하고 동일 task gate에서 효용을 검증하는 것입니다.</p>
      <div class='diagram'>{architecture_svg}</div>
      <div class='callout'><b>두 개의 truth를 분리합니다.</b> native multi-exposure RAW는 exposure·gain·CFA·black/white·calibration의 source attestation이 있을 때 radiometric/HDR claim 후보가 됩니다. nuScenes RGB → CameraE2E 경로는 동일 RGB scene의 forward re-capture로 만든 pseudo-RAW이며, 파이프라인·metadata·temporal risk를 검증하지만 native RAW 복원 또는 실제 bracket 품질을 주장하지 않습니다.</div>
      <h3>입력 계약과 provenance gate</h3><div class='table-wrap'><table><thead><tr><th>계층</th><th>계약</th><th>실패를 막는 이유</th></tr></thead><tbody>
      {_table_row('Source / capture','native_sensor · simulated_multi_exposure · nuscenes_processed_jpeg_pseudo_raw를 배타적으로 기록','JPEG proxy를 native RAW/HDR evidence로 잘못 승격하는 것을 차단',esc=esc)}
      {_table_row('RawFrame','E×H×W exposure-first RAW, SensorMetadata, CalibrationProfile, provenance','pixel array와 metadata/calibration 출처가 분리되어 drift하는 것을 방지',esc=esc)}
      {_table_row('metadata_field_origins','source_dataset / simulator_configured / simulator_readback / bridge_assumed / unknown','값이 채워져 있다는 사실과 실제 측정·readback 근거를 구분',esc=esc)}
      {_table_row('PreviousFrameState','직전 linear perception luma/RGB, timestamp, frame counter','temporal residual을 명시적으로 연결; 단 same-camera 검증은 현재 caller 책임',esc=esc)}
      </tbody></table></div>
      <h3>10단계 실행 순서</h3><div class='architecture-grid'>
      <div class='stage'><b>1 · Sensor interface</b>shape·finite·exposure count·CFA·readout·provenance 모순을 검사하고 exposure-first로 정규화합니다.</div>
      <div class='stage'><b>2 · Physical normalization</b>companding, black/white, FPN/DSNU, analog·digital gain, PRNU, lens shading, defect 보정을 적용합니다.</div>
      <div class='stage'><b>3 · HDR fusion</b>노출시간 radiance proxy와 SNR·포화·저신호·plane 일치도 weight로 합성하고 source·ghost risk를 남깁니다.</div>
      <div class='stage'><b>4 · CFA decoder</b>Bayer edge-aware/bilinear, RCC*, RGB-IR, mono별 measured sample과 interpolation 경로를 분리합니다.</div>
      <div class='stage'><b>5 · Noise engine</b>shot/read/dark/quantization/calibration residual과 lens gain을 variance·SNR로 변환합니다.</div>
      <div class='stage'><b>6 · Edge/optics</b>Y, R−G, B−G의 pixel-local structure tensor와 MTF/PSF prior로 strength·orientation·confidence를 만듭니다.</div>
      <div class='stage'><b>7 · Color/spectral</b>camera/perception matrix, log chromaticity, WB-balance heuristic, IR contamination을 계산합니다.</div>
      <div class='stage'><b>8 · Geometry/timing</b>optional nearest dewarp, row timestamps, rolling-readout phase, calibration matrices를 전달합니다.</div>
      <div class='stage'><b>9 · Temporal/flicker</b>이전 luma와의 photometric residual, exponential consistency, bright/flicker risk를 계산합니다.</div>
      <div class='stage'><b>10 · Image formation</b>edge/demosaic-aware suppression·denoise·tone map으로 Human RGB와 Vision RGB를 만들고 tensor/health/trace를 포맷합니다.</div>
      </div>
      <div class='callout warning'><b>코드 감사로 바로잡은 점:</b> lens gain은 HDR 이후가 아니라 normalization 단계에서 이미 적용됩니다. WB gain은 현재 추정·metadata로만 남고 RGB에 실제 적용되지 않습니다. temporal map도 현재 image formation RGB를 바꾸지 않습니다. 따라서 이 보고서는 기존 개념도보다 실제 호출·계산 순서를 우선합니다.</div>
      <h3>결과물과 소비 경로</h3><div class='table-wrap'><table><thead><tr><th>결과</th><th>내용</th><th>설계 의도·현재 경계</th></tr></thead><tbody>
      {_table_row('human_rgb','camera color path + artifact suppression + denoise + tone map','사람 중심 비교 기준; production HumanISP 동등성을 뜻하지 않음',esc=esc)}
      {_table_row('vision_rgb','perception color path + weak denoise + tone map','machine-view RGB baseline; Aux 없이도 독립 평가 가능',esc=esc)}
      {_table_row('maps','6개 기능군의 33개 H×W 또는 row map','해석 가능한 sensor evidence; class label이나 calibrated probability가 아님',esc=esc)}
      {_table_row('accurate','RGB 3 + selected Aux 18 = 21 channels','full-resolution software reference tensor',esc=esc)}
      {_table_row('fast','ROI의 luma/edge strength/edge confidence/temporal difference/saturation/noise = 6 channels','현재 full pipeline 이후 생성되므로 진짜 streaming early path가 아님',esc=esc)}
      {_table_row('DNN profiles','stable RGB+Aux 6 channels / extended 16 channels','모든 33개를 무조건 학습 입력으로 쓰지 않고 목적별 subset을 사용',esc=esc)}
      {_table_row('health · runtime · trace','scene health, 다음 설정 제안, 11개 opt-in intermediate arrays','제안은 closed loop가 아니며 latency는 모델 추정값',esc=esc)}
      </tbody></table></div>
      <h3>RGB+Aux 학습 결합</h3><div class='formula-strip'><div class='formula-box'><b>두 branch</b><code>R = Conv_RGB(RGB)<br>A = BN_aux(Conv_Aux(Aux))</code><span>pretrained RGB 표현과 sensor evidence를 분리해 초기화·freeze·ablation이 가능합니다.</span></div><div class='formula-box'><b>보수적 gated fusion</b><code>F₀ = R + sigmoid(g) · A<br>F = Act(BN(F₀))</code><span>g=-2 초기값이면 Aux 기여가 약 0.119에서 시작해 RGB baseline 훼손 위험을 낮춥니다.</span></div><div class='formula-box'><b>Claim gate</b><code>Δtask = Metric(RGB+Aux) − Metric(RGB)<br>same split/model/budget/seed protocol</code><span>map의 직관이 아니라 held-out task gain·FP·CI·counterexample으로 채택 여부를 결정합니다.</span></div></div>
      <h3>왜 이 구조인가</h3><ul class='audit-list'><li><b>정보 보존:</b> RGB formation에서 버려지는 exposure source, saturation, noise, CFA support, optics, timing을 side information으로 남깁니다.</li><li><b>공간 정렬:</b> 대부분의 Aux를 RGB와 같은 좌표계의 H×W map으로 만들어 early fusion과 pixel-level audit이 가능합니다.</li><li><b>단조·해석 가능:</b> risk가 커질수록 confidence가 낮아지는 bounded gate를 우선해 실패 방향을 추적할 수 있습니다.</li><li><b>모듈성:</b> calibration passthrough, physics-inspired model, project heuristic을 구분해 sensor profile과 task profile을 독립 교체할 수 있습니다.</li><li><b>반증 가능성:</b> 각 map은 적용 조건과 neutral fallback을 명시하고 group ablation·zero-Aux·matched RGB baseline으로 효용을 반증할 수 있습니다.</li><li><b>제품 경계:</b> camera-side low-level evidence까지만 담당하고 object semantics·BEV·planning은 downstream에 남깁니다.</li></ul>
    </section>
    <section class='tabpanel' id='aux' role='tabpanel' aria-labelledby='tab-aux' hidden>
      <h2>3. 33개 Aux map: 실패 상황 → 수식 → 기대 효과</h2><p class='lead'>아래 33개 항목은 이름만 붙인 시각화가 아닙니다. public map contract, 실제 구현식, 선택 근거, 적용 조건을 같은 카드에 묶었으며 catalog와 이름 집합이 달라지면 테스트가 실패하도록 동기화했습니다.</p>
      <div class='aux-principles'><div class='principle'><b>선정 기준 1 · 관측 가능성</b>RAW plane, metadata, calibration, 이전 state에서 계산 가능한 low-level evidence만 포함합니다. object class 같은 semantic label은 Aux가 아닙니다.</div><div class='principle'><b>선정 기준 2 · 상보성</b>단순 RGB가 숨기는 exposure·clipping·noise·CFA·optics·timing 실패축을 우선합니다. 서로 비슷한 map은 ablation 대상으로 남깁니다.</div><div class='principle'><b>선정 기준 3 · 감사 가능성</b>수식·값 의미·neutral fallback·소비 경로를 설명할 수 있어야 하며 최종 채택은 matched task evidence로 결정합니다.</div></div>
      <div class='callout warning'><b>근거 수준을 혼동하지 않습니다.</b> Poisson–Gaussian noise, exposure normalization, structure tensor, calibration transfer는 물리·고전 추정 원리에서 출발합니다. 여러 항을 곱하거나 threshold로 묶은 confidence/flicker/likelihood는 이 프로젝트의 명시적 heuristic이며 fitted probability가 아닙니다.</div>
      {aux_rationale_html}
    </section>
    <section class='tabpanel' id='status' role='tabpanel' aria-labelledby='tab-status' hidden>
      <h2>4. 현재 진행상황</h2><div class='cards'><div class='card'><b>{esc(showcase.get('source_frames'))}</b><span>nuScenes source frames</span></div><div class='card'><b>{esc(showcase.get('pipeline_results'))}</b><span>fresh pipeline results</span></div><div class='card'><b>{esc(showcase.get('aux_maps_per_result'))}</b><span>Aux maps / result</span></div><div class='card'><b>{esc(showcase.get('aux_map_artifacts'))}</b><span>Aux PNG artifacts</span></div><div class='card'><b>{esc(test_value)}</b><span>full regression verification</span></div><div class='card'><b>{esc(validation.get('camerae2e_python312_runtime_smoke'))}</b><span>CameraE2E Python 3.12 smoke</span></div></div>
      <h3>Milestone 상태</h3><div class='table-wrap'><table><thead><tr><th>영역</th><th>상태</th><th>완료된 것</th><th>남은 것</th></tr></thead><tbody>{milestone_rows}</tbody></table></div>
      <h3>Repository 상태</h3><div class='grid2'><div class='panel'><b>PerceptionISP</b><p>HEAD <code>{esc(p_git.get('head'))}</code> · dirty entries {esc(p_git.get('dirty_count'))}</p><p class='small'>{esc(p_git.get('path'))}</p></div><div class='panel'><b>CameraE2E</b><p>HEAD <code>{esc(c_git.get('head'))}</code> · dirty entries {esc(c_git.get('dirty_count'))}</p><p class='small'>{esc(c_git.get('path'))}</p></div></div>
      <div class='callout danger'><b>현재 반드시 정리해야 할 점:</b> 두 repository 모두 미커밋 변경을 포함합니다. 기능은 검증됐지만, 재현 가능한 기준점으로 사용하려면 변경 범위를 review하고 의도적으로 commit한 뒤 clean checkout에서 acceptance를 다시 실행해야 합니다.</div>
    </section>
    <section class='tabpanel' id='evidence' role='tabpanel' aria-labelledby='tab-evidence' hidden>
      <h2>5. 검증된 성과와 근거 수준</h2><div class='table-wrap'><table><thead><tr><th>주장</th><th>근거</th><th>판정</th><th>경계</th></tr></thead><tbody>
      {_table_row('Software feasibility','RAW/CFA ISP, CameraE2E bridge, native loader, 33 maps, training/evaluation/report path','Supported','Production quality 또는 task superiority와는 별개',esc=esc)}
      {_table_row('Native RAW FP/precision signal',f"PascalRAW {native.get('sample_count')} samples: dP50={_fmt(native.get('precision50_delta'))}, dFP50={_fmt(native.get('fp50_delta'))}, dR50={_fmt(native.get('recall50_delta'))}",'Supported in scoped gate','RGGB-heavy 단일 protocol; recall trade-off 존재',esc=esc)}
      {_table_row('DNN이 Aux 입력을 실제 사용',f"Compact dense, {dense.get('seed_count')} seeds; zero-Aux recall delta={_fmt(dense.get('zero_aux_recall_delta'))}",'Supported','Production detector 또는 adverse benchmark 증거는 아님',esc=esc)}
      {_table_row('Edge evidence 방향성',f"scene edge best={scene.get('aux_best_name')}, dF1={_fmt(scene.get('aux_best_d_f1'))}",'Diagnostic positive','Object contour GT가 아닌 proxy 포함',esc=esc)}
      {_table_row('YOLO broad gain',f"seed2 dR={_fmt(yolo.get('recall_delta'))}, dAP50-95={_fmt(yolo.get('map5095_delta'))}",'Not broadly supported','Recall/AP trade-off와 이전 optimizer-path 문제',esc=esc)}
      {_table_row('Corrected hard-slice signal',f"all dAP50-95={_fmt(hard_all.get('map5095_delta'))}, small={_fmt(hard_small.get('map5095_delta'))}, thin={_fmt(hard_thin.get('map5095_delta'))}",'Promising but partial','seed3/4 동일 반복은 독립 variance 3-seed로 계산 불가',esc=esc)}
      {_table_row('nuScenes CameraE2E showcase',f"{showcase.get('pipeline_results')} results, {showcase.get('aux_map_artifacts')} maps, cache hits={showcase.get('cache_hits')}",'Contract/evidence pass','pseudo-RAW이며 native HDR quality claim=false',esc=esc)}
      </tbody></table></div>
      <h3>Evidence ladder</h3><ol class='checklist'><li><b>Unit/mechanism:</b> 알고리즘과 계약이 기대 방향으로 동작하는가?</li><li><b>Simulator/data bridge:</b> CameraE2E와 nuScenes metadata/provenance가 올바른가?</li><li><b>Native sensor:</b> 실제 RAW와 calibration에서 물리적으로 유효한가?</li><li><b>Task benchmark:</b> 동일 예산의 RGB baseline보다 held-out 성능이 개선되는가?</li><li><b>Production:</b> 지연·전력·메모리·고정소수점·안전 요구를 만족하는가?</li></ol><p>현재는 1–2가 강하고, 3은 제한된 protocol, 4는 부분적·혼합 결과, 5는 아직 open입니다.</p>
    </section>
    <section class='tabpanel' id='limits' role='tabpanel' aria-labelledby='tab-limits' hidden>
      <h2>6. 현재 한계와 위험</h2><div class='grid2'><div class='panel'><h3>물리·데이터 한계</h3><ul class='checklist'><li>nuScenes는 processed auto-exposed JPEG이며 source exposure/gain/WB/CFA/black·white가 없습니다.</li><li>CameraE2E는 forward recapture simulator이고 inverse ISP나 JPEG clipping 복원기가 아닙니다.</li><li>인접-frame HDR 근사는 registration·pose warp·optical flow·deghosting이 없는 stress stack입니다.</li><li>기본 MTF/PSF/lens/noise map은 실센서 calibration을 대신하지 못합니다.</li><li><b>origin이 unknown인 metadata도 현재 기본 숫자로 계산될 수 있습니다.</b> 따라서 값의 존재와 물리적 calibration validity를 분리하는 gate가 필요합니다.</li><li>일부 native loader는 실제 mosaic/CFA를 읽어도 source exposure·gain을 완전히 복원하지 못하고 기본값을 사용합니다. “native CFA”와 “완전 calibrated RAW”는 같은 뜻이 아닙니다.</li><li>CameraE2E의 독립 호출 noise/FPN persistence는 실센서 temporal noise로 calibration되지 않았습니다.</li><li>HDR noise variance는 fusion weight별 분산을 엄밀히 전파하지 않고 fused signal에 단일 모델을 다시 적용하며, dark-current 항은 첫 exposure time만 사용합니다.</li><li><code>saturation</code>과 <code>clipping_distance</code>는 fused recoverability가 아니라 any/max input-plane 기준이라 짧은 노출이 복구한 위치에도 보수적으로 반응할 수 있습니다.</li></ul></div><div class='panel'><h3>성능·제품화 한계</h3><ul class='checklist'><li>HumanISP 대비 보편적 detector 우월성이 증명되지 않았습니다.</li><li>33개 map 전체가 필요한지, 어떤 subset이 task별 최적인지 미확정입니다.</li><li><b>현재 fast tensor는 전체 HDR·demosaic·color·formation 뒤에 만들어집니다.</b> 따라서 실제 streaming early-output path가 아닙니다.</li><li>Fast-path latency는 모델 추정치이고 실제 hardware 측정치가 아닙니다.</li><li>현재 320×180 showcase bundle은 전체 lossless trace 때문에 약 1.1GB로 큽니다.</li><li>float64 software reference를 embedded fixed-point ISP로 옮기는 설계는 아직 없습니다.</li><li>RuntimeController는 다음 설정을 제안하지만 closed-loop로 재적용하지 않습니다.</li><li><b>감사에서 발견:</b> extended tensor는 16채널인데 feature-distillation loader의 CHW/HWC 판별 집합에 16이 없어 <code>16×H×W</code> NPY를 오해석할 수 있습니다. 수정·회귀 테스트 전에는 해당 조합을 claim run에 사용하면 안 됩니다.</li><li><code>output_bit_depth</code>, <code>edge_confidence_floor</code>, <code>include_raw_like_tensor</code>, <code>color_shading_gain</code>은 현재 core processing에서 실질적으로 소비되지 않습니다.</li></ul></div></div>
      <div class='callout danger'><b>금지해야 할 표현:</b> “nuScenes JPEG가 native RAW로 복원됐다”, “인접 JPEG가 실제 HDR bracket이다”, “ghost map이 deghosting을 완료했다”, “33개 Aux가 detector 성능을 자동 향상한다”, “현재 latency가 production hardware 성능이다”.</div>
    </section>
    <section class='tabpanel' id='next' role='tabpanel' aria-labelledby='tab-next' hidden>
      <h2>7. 다음 해야 할 일</h2><p>아래 순서는 기능을 더 많이 추가하는 순서가 아니라, 현재 주장의 가장 큰 불확실성을 제거하는 순서입니다.</p><div class='table-wrap'><table><thead><tr><th>우선순위</th><th>작업</th><th>왜 필요한가</th><th>Acceptance 기준</th><th>선행조건</th></tr></thead><tbody>{roadmap_rows}</tbody></table></div>
      <h3>권장 실행 순서</h3><div class='flow'><div class='node'><strong>P0-A. 기준점 고정</strong>두 repo review·commit, 환경 lock, clean CI acceptance.</div><div class='node'><strong>P0-B. Native truth</strong>실제 multi-exposure RAW와 sensor calibration 확보.</div><div class='node'><strong>P0-C. Fair task gate</strong>진짜 독립 3+ seeds, matched RGB baseline, CI와 hard slices.</div><div class='node'><strong>P1-A. Temporal HDR</strong>품질 주장이 필요할 때만 registration와 deghosting 구현.</div><div class='node'><strong>P1-B. Aux 축소</strong>task profile과 ablation으로 33개를 제품 입력 subset으로 정리.</div><div class='node'><strong>P1-C/P2. Productization</strong>float32·압축·실측 latency·fixed point·on-device 검증.</div></div>
      <div class='callout warning'><b>중요한 선택:</b> 목표가 “temporal risk 진단”이면 현재 unregistered stack이 충분히 의미 있습니다. 목표가 “실제 HDR 화질 개선”이면 registration/deghosting/native RAW가 필수입니다. 두 목표를 한 보고서 문구로 섞으면 안 됩니다.</div>
    </section>
    <section class='tabpanel' id='use' role='tabpanel' aria-labelledby='tab-use' hidden>
      <h2>8. 실행 방법과 근거 자료</h2><h3>대표 명령</h3><div class='panel'><p><b>기본 ISP</b><br><code>perception-isp isp run --width 320 --height 180 --cfa RGGB</code></p><p><b>nuScenes 전체 showcase</b><br><code>perception-isp report showcase data/nuscenes --scene-name scene-0061 --camera CAM_FRONT --frames 6</code></p><p><b>Native HDR</b><br><code>perception-isp example native-hdr DATA/frame_000.npz</code></p><p><b>이 상태 보고서 재생성</b><br><code>perception-isp report status --project-root .</code></p></div>
      <h3>관련 보고서·문서</h3><div class='links'><a href='../perception_nuscenes_scene0061_camerae2e_pipeline_showcase_v1/index.html'>nuScenes–CameraE2E pipeline showcase</a><a href='../perception_project_accomplishment_tabs_current_v1/index.html'>Historical accomplishment report</a><a href='../../docs/ARCHITECTURE.md'>Architecture</a><a href='../../docs/EVIDENCE_AND_LIMITATIONS.md'>Evidence & limitations</a><a href='../../docs/USER_GUIDE_KO.md'>Korean user guide</a><a href='../../src/perception_isp/core/aux_map_catalog.py'>Aux public contract</a><a href='../../src/perception_isp/core/aux_map_rationale.py'>Aux equations/rationale</a><a href='project_status_summary.json'>Machine-readable summary</a></div>
      <h3>설계 원리의 1차 근거</h3><div class='table-wrap'><table><thead><tr><th>원리</th><th>이 프로젝트에서 차용한 부분</th><th>그대로 같지 않은 부분</th></tr></thead><tbody>
      <tr><td><a href='https://www.pauldebevec.com/Research/HDR/'>Debevec &amp; Malik, HDR radiance maps</a></td><td>서로 다른 노출을 exposure-aware radiance 표현으로 결합한다는 원리</td><td>현재 weight·motion penalty·fallback은 PerceptionISP의 명시적 heuristic이며 response-curve recovery 구현이 아님</td></tr>
      <tr><td><a href='https://doi.org/10.1109/TIP.2008.2001399'>Foi et al., Poisson–Gaussian RAW noise</a></td><td>signal-dependent shot component와 additive noise component를 분산으로 모델링</td><td>현재 coefficient는 profile 의존이며 HDR weight variance propagation은 근사</td></tr>
      <tr><td><a href='https://www.bmva-archive.org.uk/bmvc/1988/avc-88-023.html'>Harris &amp; Stephens, edge/corner second moment</a></td><td>gradient second-moment eigenvalue·anisotropy·orientation 해석</td><td>현재 J는 별도 spatial-window smoothing 없는 pixel-local, noise-weighted 3-channel form</td></tr>
      <tr><td><a href='https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html'>nuScenes primary paper</a></td><td>시간·pose·calibration이 있는 multi-sensor sequence를 temporal contract에 사용</td><td>제공 camera image는 native multi-exposure RAW가 아니므로 HDR radiometry claim에는 사용하지 않음</td></tr>
      <tr><td><a href='https://www.nuscenes.org/terms-of-use'>nuScenes Dataset Terms</a></td><td>공개 report snapshot의 파생 PNG는 report-local notice와 CC BY-NC-SA 4.0 조건으로 배포</td><td>원본 nuScenes JPEG·테이블과 lossless NPY trace는 GitHub snapshot에 포함하지 않음</td></tr>
      </tbody></table></div>
      <h3>환경 조건</h3><div class='table-wrap'><table><tbody>{_table_row('PerceptionISP core',software.get('core_python'),esc=esc)}{_table_row('CameraE2E runtime',software.get('camerae2e_python'),esc=esc)}{_table_row('Claim command fallback','허용하지 않음',esc=esc)}{_table_row('Provenance categories','source_dataset / simulator_configured / simulator_readback / bridge_assumed / unknown',esc=esc)}</tbody></table></div>
    </section>
    </main><script>(()=>{{const tabs=[...document.querySelectorAll('[role="tab"]')],panels=[...document.querySelectorAll('[role="tabpanel"]')];function activate(id,hash=true){{if(!panels.some(p=>p.id===id))id='purpose';tabs.forEach(t=>{{const on=t.dataset.tab===id;t.setAttribute('aria-selected',String(on));t.tabIndex=on?0:-1}});panels.forEach(p=>p.hidden=p.id!==id);if(hash)history.replaceState(null,'','#'+id);window.scrollTo({{top:document.querySelector('.tabs').offsetTop-6,behavior:'smooth'}})}}tabs.forEach((t,i)=>{{t.addEventListener('click',()=>activate(t.dataset.tab));t.addEventListener('keydown',e=>{{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();let n=e.key==='Home'?0:e.key==='End'?tabs.length-1:e.key==='ArrowLeft'?(i-1+tabs.length)%tabs.length:(i+1)%tabs.length;tabs[n].focus();activate(tabs[n].dataset.tab)}})}});activate(location.hash.slice(1)||'purpose',false);addEventListener('hashchange',()=>activate(location.hash.slice(1),false))}})();</script></body></html>"""


def _render_aux_rationales(esc: Any) -> str:
    group_info = {
        "normalization_calibration": (
            "Normalization & calibration",
            "센서 불균일성과 보정 이력을 숨기지 않고, 보정된 값의 신뢰도와 noise 증폭까지 함께 전달합니다.",
        ),
        "hdr_exposure": (
            "HDR & exposure",
            "exposure-time radiance proxy, SNR, clipping, low-signal, plane disagreement를 분리해 fusion 결과의 출처와 실패 위험을 감사합니다.",
        ),
        "cfa_spectral": (
            "CFA, color & spectral",
            "측정 CFA sample과 보간 color를 구분하고, 밝기·색도·IR/clear channel의 신뢰 조건을 보존합니다.",
        ),
        "noise_reliability": (
            "Noise & reliability",
            "Poisson–Gaussian-inspired variance와 표준화된 SNR/gradient로 signal과 uncertainty를 같은 좌표에서 제공합니다.",
        ),
        "edge_optics": (
            "Edge & optics",
            "noise-weighted pixel-local second-moment tensor에 MTF/PSF prior를 결합해 gradient의 세기와 신뢰를 분리합니다.",
        ),
        "timing_temporal": (
            "Timing & temporal",
            "rolling-readout phase와 이전 frame photometric residual을 노출해 motion/flicker 위험을 진단합니다.",
        ),
    }
    groups: dict[str, list[Any]] = {}
    for spec in AUX_MAP_SPECS:
        groups.setdefault(spec.group, []).append(spec)
    rendered: list[str] = []
    for group_index, (group, specs) in enumerate(groups.items()):
        label, intro = group_info.get(group, (group.replace("_", " / "), ""))
        cards: list[str] = []
        for spec in specs:
            rationale = AUX_MAP_RATIONALE_BY_NAME[spec.name]
            consumers = ", ".join(spec.consumers) if spec.consumers else "diagnostic / optional consumer"
            cards.append(
                "<article class='aux-card' data-aux-name='{}'>"
                "<h3><code>{}</code><span class='tag'>{}</span></h3>"
                "<div class='small'>stage <code>{}</code> · consumer {}</div>"
                "<h4>해결하려는 상황</h4><p>{}</p>"
                "<h4>실제 구현 수식</h4><pre>{}</pre>"
                "<h4>선정한 알고리즘과 근거</h4><p>{}</p>"
                "<h4>왜 도움이 될 수 있는가</h4><p>{}</p>"
                "<h4>값의 의미</h4><p>{}</p>"
                "<h4>해석 경계</h4><p class='boundary'>{}</p>"
                "<div class='small'>implementation <code>{}</code></div>"
                "</article>".format(
                    esc(spec.name),
                    esc(spec.name),
                    esc(spec.derivation),
                    esc(spec.stage),
                    esc(consumers),
                    esc(rationale.problem_situation),
                    esc(rationale.formula),
                    esc(rationale.design_basis),
                    esc(rationale.why_it_helps),
                    esc(spec.value_semantics),
                    esc(rationale.interpretation_boundary),
                    esc(spec.implementation_ref),
                )
            )
        open_attribute = " open" if group_index == 0 else ""
        rendered.append(
            f"<details class='aux-group'{open_attribute}><summary>{esc(label)} · {len(specs)} maps</summary>"
            f"<p class='aux-group-intro'>{esc(intro)}</p><div class='aux-grid'>{''.join(cards)}</div></details>"
        )
    return "".join(rendered)


def _architecture_svg() -> str:
    return """<svg viewBox='0 0 1450 720' role='img' aria-labelledby='architecture-title architecture-desc'>
      <title id='architecture-title'>PerceptionISP end-to-end architecture</title>
      <desc id='architecture-desc'>Native RAW or nuScenes RGB recaptured by CameraE2E enters a provenance-aware RawFrame, passes through ten explicit ISP stages, then produces RGB, auxiliary maps, tensors, health, trace, and matched RGB versus RGB plus Aux task evidence.</desc>
      <defs>
        <marker id='arrowhead' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' fill='#58766d'/></marker>
        <marker id='arrowhead-purple' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'><path d='M0,0 L8,4 L0,8 z' fill='#8b6ab3'/></marker>
      </defs>
      <text x='20' y='18' class='title'>CAPTURE &amp; CONTRACT</text>
      <rect x='20' y='32' width='170' height='62' rx='10' class='box source'/><text x='105' y='57' text-anchor='middle' class='title'>nuScenes RGB</text><text x='105' y='76' text-anchor='middle' class='sub'>processed JPEG · temporal source</text>
      <rect x='215' y='32' width='205' height='62' rx='10' class='box source'/><text x='318' y='56' text-anchor='middle' class='title'>CameraE2E</text><text x='318' y='75' text-anchor='middle' class='sub'>forward re-capture · AE/AEB</text>
      <rect x='447' y='32' width='188' height='62' rx='10' class='box source'/><text x='541' y='56' text-anchor='middle' class='title'>pseudo-RAW E×H×W</text><text x='541' y='75' text-anchor='middle' class='sub'>simulated provenance only</text>
      <rect x='447' y='111' width='188' height='62' rx='10' class='box source'/><text x='541' y='135' text-anchor='middle' class='title'>Native RAW E×H×W</text><text x='541' y='154' text-anchor='middle' class='sub'>attested sensor path</text>
      <rect x='690' y='64' width='205' height='76' rx='12' class='box core'/><text x='793' y='91' text-anchor='middle' class='title'>RawFrame contract</text><text x='793' y='110' text-anchor='middle' class='sub'>RAW + metadata + calibration</text><text x='793' y='126' text-anchor='middle' class='sub'>+ field-level provenance</text>
      <rect x='960' y='64' width='445' height='76' rx='12' class='box gate'/><text x='1183' y='90' text-anchor='middle' class='title'>Origin / contradiction gate</text><text x='1183' y='110' text-anchor='middle' class='sub'>source_dataset · simulator_configured · simulator_readback</text><text x='1183' y='126' text-anchor='middle' class='sub'>bridge_assumed · unknown · native/pseudo exclusivity</text>
      <path d='M190 63 H215' class='wire'/><path d='M420 63 H447' class='wire'/><path d='M635 63 H690' class='wire'/><path d='M635 142 H665 Q690 142 690 117' class='wire'/><path d='M960 102 H920 Q895 102 895 102' class='sidewire'/>

      <text x='20' y='205' class='title'>EXPLICIT SOFTWARE ISP · ACTUAL CALL ORDER</text>
      <rect x='20' y='222' width='126' height='72' rx='10' class='box core'/><text x='83' y='247' text-anchor='middle' class='title'>1 Interface</text><text x='83' y='266' text-anchor='middle' class='sub'>shape · CFA</text><text x='83' y='281' text-anchor='middle' class='sub'>finite · origins</text>
      <rect x='160' y='222' width='142' height='72' rx='10' class='box core'/><text x='231' y='247' text-anchor='middle' class='title'>2 Normalize</text><text x='231' y='266' text-anchor='middle' class='sub'>levels · gains</text><text x='231' y='281' text-anchor='middle' class='sub'>PRNU · lens · defect</text>
      <rect x='316' y='222' width='126' height='72' rx='10' class='box core'/><text x='379' y='247' text-anchor='middle' class='title'>3 HDR</text><text x='379' y='266' text-anchor='middle' class='sub'>radiance weights</text><text x='379' y='281' text-anchor='middle' class='sub'>source · ghost risk</text>
      <rect x='456' y='222' width='126' height='72' rx='10' class='box core'/><text x='519' y='247' text-anchor='middle' class='title'>4 CFA</text><text x='519' y='266' text-anchor='middle' class='sub'>Bayer · RCC*</text><text x='519' y='281' text-anchor='middle' class='sub'>RGB-IR · mono</text>
      <rect x='596' y='222' width='126' height='72' rx='10' class='box core'/><text x='659' y='247' text-anchor='middle' class='title'>5 Noise</text><text x='659' y='266' text-anchor='middle' class='sub'>variance · SNR</text><text x='659' y='281' text-anchor='middle' class='sub'>normalized ∇</text>
      <rect x='736' y='222' width='142' height='72' rx='10' class='box core'/><text x='807' y='247' text-anchor='middle' class='title'>6 Edge/optics</text><text x='807' y='266' text-anchor='middle' class='sub'>local tensor</text><text x='807' y='281' text-anchor='middle' class='sub'>MTF · PSF gates</text>
      <rect x='892' y='222' width='126' height='72' rx='10' class='box core'/><text x='955' y='247' text-anchor='middle' class='title'>7 Color</text><text x='955' y='266' text-anchor='middle' class='sub'>matrices · ratios</text><text x='955' y='281' text-anchor='middle' class='sub'>WB estimate only</text>
      <rect x='1032' y='222' width='126' height='72' rx='10' class='box core'/><text x='1095' y='247' text-anchor='middle' class='title'>8 Geometry</text><text x='1095' y='266' text-anchor='middle' class='sub'>optional dewarp</text><text x='1095' y='281' text-anchor='middle' class='sub'>row timing</text>
      <rect x='1172' y='222' width='126' height='72' rx='10' class='box core'/><text x='1235' y='247' text-anchor='middle' class='title'>9 Temporal</text><text x='1235' y='266' text-anchor='middle' class='sub'>photometric Δ</text><text x='1235' y='281' text-anchor='middle' class='sub'>flicker risk</text>
      <rect x='1312' y='222' width='118' height='72' rx='10' class='box core'/><text x='1371' y='247' text-anchor='middle' class='title'>10 Form</text><text x='1371' y='266' text-anchor='middle' class='sub'>suppress · denoise</text><text x='1371' y='281' text-anchor='middle' class='sub'>tone map</text>
      <path d='M793 140 V185 H83 V222' class='wire'/><path d='M146 258 H160' class='wire'/><path d='M302 258 H316' class='wire'/><path d='M442 258 H456' class='wire'/><path d='M582 258 H596' class='wire'/><path d='M722 258 H736' class='wire'/><path d='M878 258 H892' class='wire'/><path d='M1018 258 H1032' class='wire'/><path d='M1158 258 H1172' class='wire'/><path d='M1298 258 H1312' class='wire'/>
      <rect x='1165' y='158' width='140' height='43' rx='9' class='box gate'/><text x='1235' y='176' text-anchor='middle' class='title'>Previous state</text><text x='1235' y='192' text-anchor='middle' class='sub'>caller-chained luma</text><path d='M1235 201 V222' class='sidewire'/>

      <text x='20' y='342' class='title'>OUTPUT CONTRACTS</text>
      <rect x='20' y='360' width='170' height='70' rx='10' class='box output'/><text x='105' y='385' text-anchor='middle' class='title'>Human RGB</text><text x='105' y='405' text-anchor='middle' class='sub'>display-oriented comparator</text>
      <rect x='210' y='360' width='170' height='70' rx='10' class='box output'/><text x='295' y='385' text-anchor='middle' class='title'>Vision RGB</text><text x='295' y='405' text-anchor='middle' class='sub'>machine-view baseline</text>
      <rect x='400' y='360' width='170' height='70' rx='10' class='box output'/><text x='485' y='385' text-anchor='middle' class='title'>33 Aux maps</text><text x='485' y='405' text-anchor='middle' class='sub'>6 failure-mode groups</text>
      <rect x='590' y='360' width='170' height='70' rx='10' class='box output'/><text x='675' y='385' text-anchor='middle' class='title'>Accurate 21ch</text><text x='675' y='405' text-anchor='middle' class='sub'>full-frame RGB + 18 Aux</text>
      <rect x='780' y='360' width='170' height='70' rx='10' class='box output'/><text x='865' y='385' text-anchor='middle' class='title'>Fast 6ch</text><text x='865' y='405' text-anchor='middle' class='sub'>stripe · estimated latency</text>
      <rect x='970' y='360' width='170' height='70' rx='10' class='box output'/><text x='1055' y='385' text-anchor='middle' class='title'>Health / control</text><text x='1055' y='405' text-anchor='middle' class='sub'>suggestion, not closed loop</text>
      <rect x='1160' y='360' width='170' height='70' rx='10' class='box output'/><text x='1245' y='385' text-anchor='middle' class='title'>Trace / next state</text><text x='1245' y='405' text-anchor='middle' class='sub'>opt-in arrays · temporal link</text>
      <path d='M1371 294 V325 H105 V360' class='wire'/><path d='M1371 325 H295 V360' class='wire'/><path d='M1235 294 V335 H485 V360' class='wire'/><path d='M1371 325 H675 V360' class='wire'/><path d='M1235 335 H865 V360' class='wire'/><path d='M955 294 V340 H1055 V360' class='wire'/><path d='M1235 294 V360' class='wire'/>

      <text x='20' y='487' class='title'>LEARNED USE &amp; EVIDENCE GATE</text>
      <rect x='160' y='510' width='250' height='78' rx='12' class='box gate'/><text x='285' y='537' text-anchor='middle' class='title'>Selected tensor profile</text><text x='285' y='557' text-anchor='middle' class='sub'>stable 6ch or extended 16ch</text><text x='285' y='574' text-anchor='middle' class='sub'>not all 33 maps by default</text>
      <rect x='510' y='510' width='310' height='78' rx='12' class='box gate'/><text x='665' y='537' text-anchor='middle' class='title'>Gated two-branch stem</text><text x='665' y='557' text-anchor='middle' class='sub'>F = ConvRGB + sigmoid(g)·BN(ConvAux)</text><text x='665' y='574' text-anchor='middle' class='sub'>then shared BN · activation · backbone</text>
      <rect x='920' y='510' width='360' height='78' rx='12' class='box gate'/><text x='1100' y='537' text-anchor='middle' class='title'>Matched task &amp; claim gate</text><text x='1100' y='557' text-anchor='middle' class='sub'>same split · model · budget · operating point</text><text x='1100' y='574' text-anchor='middle' class='sub'>metrics + FP + CI + hard slices + counterexamples</text>
      <path d='M295 430 V475 H285 V510' class='sidewire'/><path d='M485 430 V475 H285 V510' class='sidewire'/><path d='M410 549 H510' class='wire'/><path d='M820 549 H920' class='wire'/>
      <rect x='160' y='628' width='1120' height='56' rx='12' class='box'/><text x='720' y='651' text-anchor='middle' class='title'>Evidence ladder: mechanism → simulator/bridge → native sensor → held-out task → production</text><text x='720' y='671' text-anchor='middle' class='sub'>현재 강점은 auditable software/bridge이며, native calibration·independent multi-seed task gain·measured hardware는 아직 다음 gate다.</text>
      <path d='M1100 588 V628' class='wire'/>
    </svg>"""


def _milestone_rows(
    showcase: Mapping[str, Any], accomplishment: Mapping[str, Any], camerae2e_smoke: str
) -> list[dict[str, str]]:
    return [
        {"area": "RAW/metadata/calibration contract", "status": "partial", "completed": "RawFrame, exposure axis, CFA and origin validation, native/pseudo contradiction rejection.", "remaining": "Unknown origin과 계산 validity 분리, sensor-specific importer, per-channel level과 profile registry."},
        {"area": "PerceptionISP core", "status": "complete", "completed": "Human/machine RGB, accurate/fast paths, health, 33 Aux maps, opt-in trace.", "remaining": "Measured latency, fixed-point mapping, closed-loop controller."},
        {"area": "CameraE2E AE/AEB", "status": "complete" if camerae2e_smoke == "pass" else "partial", "completed": "Stateful delayed AE, same-scene AE bracket, AEB sequence, readback and seed audit.", "remaining": "Clean committed integration baseline and calibrated temporal noise behavior."},
        {"area": "nuScenes temporal/showcase", "status": "complete" if showcase.get("status") == "pass" else "partial", "completed": "6-frame CAM_FRONT chain, pose/intrinsic propagation, 11-result/363-map fresh evidence bundle.", "remaining": "No native source exposure metadata; no geometric registration."},
        {"area": "Inter-frame HDR approximation", "status": "stress-only", "completed": "Unregistered dynamic stack and same-frame comparator expose ghost/flicker risk.", "remaining": "Registration, deghosting and native RAW before HDR-quality claims."},
        {"area": "Native HDR input", "status": "partial", "completed": "Strict NPZ+JSON contract and provenance contradiction checks.", "remaining": "Representative real multi-exposure sensor data and radiometric quality benchmark."},
        {"area": "RGB+Aux learned path", "status": "partial", "completed": "Compact DNN, YOLO gated fusion, distillation, corrected optimizer/freeze path.", "remaining": "Genuinely independent multi-seed win and adverse-condition generalization."},
        {"area": "Evaluation/reporting", "status": "complete", "completed": "Matched metrics, hard slices, claim gates, counterexamples, tabbed evidence reports.", "remaining": "Automated report-size optimization and CI-published acceptance summary."},
        {"area": "Production readiness", "status": "open", "completed": "Software reference only.", "remaining": "Throughput, memory, power, fixed point, safety, on-device integration."},
    ]


def _roadmap_rows() -> list[dict[str, str]]:
    return [
        {"priority": "P0", "work": "두 repository 기준점 고정", "why": "현재 PerceptionISP와 CameraE2E 모두 미커밋 변경을 포함해 재현 기준점이 불안정합니다.", "acceptance": "변경 review·commit, clean checkout, Python 3.11 core/3.12 CameraE2E CI, full tests와 scene-0061 smoke 재통과.", "depends_on": "없음"},
        {"priority": "P0", "work": "감사에서 발견한 correctness gap 수정", "why": "16채널 extended CHW tensor를 feature-distillation loader가 HWC로 오해할 수 있고, same-camera temporal state·MTF finite/range가 내부에서 강제되지 않습니다.", "acceptance": "layout을 tensor contract로 명시하고 16×H×W/H×W×16 fixture 모두 통과; PreviousFrameState camera/timestamp validation; MTF/PSF finite/range rejection 또는 명시적 clamp; regression test 추가.", "depends_on": "기준점 고정"},
        {"priority": "P0", "work": "HDR Aux 의미·분산 모델 강화", "why": "현재 any/max input saturation은 복구된 HDR 픽셀도 억제하고 noise variance는 exposure weight별 분산을 전파하지 않습니다.", "acceptance": "source-plane risk와 fused recoverability를 별도 map/semantic으로 분리; Σw²V 기반 fusion variance와 exposure별 dark-current를 synthetic radiance truth 및 native bracket에서 검증.", "depends_on": "Native truth 또는 calibrated fixture"},
        {"priority": "P0", "work": "Metadata/calibration validity gate", "why": "Origin이 unknown인 기본 수치와 실측 calibration을 구분하지 않으면 물리 Aux map을 과도하게 해석할 수 있습니다.", "acceptance": "33개 map 각각 valid/calibrated/heuristic/not-applicable 상태와 required origins 제공; unknown fixture의 radiance/noise/HDR-quality claim 차단; measured fixture 통과.", "depends_on": "기준점 고정"},
        {"priority": "P0", "work": "실제 native multi-exposure RAW 확보", "why": "현재 nuScenes path는 pseudo-RAW이므로 HDR 화질·radiance claim을 만들 수 없습니다.", "acceptance": "정적·동적 장면 E×H×W, 노출/gain/CFA/black-white/calibration source attestation, loader round-trip, pseudo/native contradiction 0건.", "depends_on": "센서 또는 공개 native HDR dataset"},
        {"priority": "P0", "work": "실센서 calibration package", "why": "noise/MTF/PSF/lens/color confidence가 neutral default가 아닌 물리적 의미를 가져야 합니다.", "acceptance": "black/white, noise, WB/CCM, lens shading, defect, MTF/PSF를 versioned profile로 저장하고 모든 field origin을 source/readback으로 감사.", "depends_on": "Native sensor와 calibration capture"},
        {"priority": "P0", "work": "공정한 task superiority gate", "why": "현재 성능 결과는 positive signal과 counterexample이 혼재합니다.", "acceptance": "동일 split/model/epoch/augmentation/운영점, 실제 stochastic source가 다른 3+ seeds, bootstrap CI, FP/sample·AP50/75·APs/m/l·boundary F1, counterexample 포함.", "depends_on": "데이터·학습 protocol freeze"},
        {"priority": "P1", "work": "Temporal registration와 deghosting", "why": "인접-frame 근사를 HDR 결과로 사용하려면 motion/ego pose 보정이 필요합니다.", "acceptance": "pose warp + optical flow residual + occlusion mask, registered/unregistered comparator, moving-edge ghost metric와 radiance error 개선.", "depends_on": "HDR-quality 목표 선택, native 또는 calibrated sequence"},
        {"priority": "P1", "work": "Aux profile/ablation 정리", "why": "33개 전체는 연구에는 유용하지만 모델 입력·대역폭·설명 가능성 측면에서 과합니다.", "acceptance": "core6/edge/HDR/temporal profile 정의, one-group-at-a-time ablation, 성능·latency·memory Pareto, stale/unused map 제거 기준.", "depends_on": "Task gate"},
        {"priority": "P1", "work": "Runtime·저장 효율화", "why": "현재 showcase는 1.1GB이며 float64 reference는 제품 배포에 비효율적입니다.", "acceptance": "float32 equivalence tolerance, compressed/chunked arrays, duplicate preview 제거, peak RAM·wall latency 측정, report size 목표 설정.", "depends_on": "Reference output freeze"},
        {"priority": "P1", "work": "RuntimeController closed loop", "why": "현재는 설정 제안만 하고 다음 capture/ISP에 자동 적용하지 않습니다.", "acceptance": "bounded state transition, hysteresis, oscillation test, exposure/flicker scenario convergence, audit log.", "depends_on": "Camera control contract"},
        {"priority": "P2", "work": "Hardware/embedded mapping", "why": "Software reference와 automotive/robotics ISP는 throughput·power·safety 요구가 다릅니다.", "acceptance": "fixed-point error budget, target SoC prototype, FPS/latency/power/memory 측정, deterministic failure handling.", "depends_on": "P0/P1 algorithm freeze"},
    ]


def _git_snapshot(path: Path) -> dict[str, Any]:
    target = Path(path)
    if not (target / ".git").exists():
        return {"available": False, "path": str(target), "head": None, "dirty_count": None}
    try:
        head = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(target), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"available": True, "path": str(target), "head": head, "dirty_count": len(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "path": str(target), "head": None, "dirty_count": None}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {"status": "invalid"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:+.4f}"


def _tab(identifier: str, label: str, selected: bool) -> str:
    return (
        f"<button class='tab' role='tab' id='tab-{identifier}' data-tab='{identifier}' "
        f"aria-controls='{identifier}' aria-selected='{'true' if selected else 'false'}'>{html.escape(label)}</button>"
    )


def _badge(value: Any, esc: Any) -> str:
    text = str(value or "unknown")
    return f"<span class='badge {esc(text)}'>{esc(text)}</span>"


def _table_row(*values: Any, esc: Any, raw_columns: set[int] | None = None) -> str:
    raw = raw_columns or set()
    cells = "".join(
        f"<td>{value if index in raw else esc(value)}</td>"
        for index, value in enumerate(values)
    )
    return f"<tr>{cells}</tr>"


if __name__ == "__main__":
    raise SystemExit(main())
