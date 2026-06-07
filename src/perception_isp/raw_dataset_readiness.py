"""Create a RAW dataset readiness report for PerceptionISP validation."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .types import json_ready


SUMMARY_FILENAME = "raw_dataset_readiness_summary.json"


DATASETS: tuple[Dict[str, Any], ...] = (
    {
        "name": "AODRaw",
        "priority": "P0",
        "recommended_use": "Primary adverse-condition real RAW object-detection benchmark.",
        "task": "object_detection",
        "domain": "indoor/outdoor adverse conditions",
        "raw_type": "real high-resolution camera RAW plus sRGB benchmark data",
        "scale": "7,785 original RAW images; train 5,445, test 2,340; 62 categories; 135,601 instances",
        "conditions": ["low_light", "rain", "fog", "mixed_light_weather", "diverse_adverse"],
        "strong_for": [
            "adverse_native_raw_slice",
            "high_information_scene",
            "task_specific_gate",
            "rgb_aux_dnn_finetune",
        ],
        "weak_for": ["automotive_only_claim"],
        "access": "official GitHub lists original, downsampled, sliced RAW/sRGB directories and COCO-format annotations",
        "source_url": "https://github.com/lzyhha/AODRaw",
        "risk": "Large download footprint; original RAW directory is hundreds of GB. Verify dataset license/access before bulk download.",
        "next_action": "Check disk budget and start with annotations plus a small downsampled or sliced RAW subset.",
    },
    {
        "name": "ROD / RAOD",
        "priority": "P0",
        "recommended_use": "Primary automotive HDR RAW benchmark if accessible.",
        "task": "object_detection",
        "domain": "day/night driving HDR scenes",
        "raw_type": "24-bit HDR RAW sensor data from Sony IMX490 Bayer sensor",
        "scale": "25k annotated RAW frames; 237k boxes; 6 driving classes",
        "conditions": ["day", "night", "strong_glare", "hdr_driving"],
        "strong_for": [
            "automotive_hdr_claim",
            "adverse_native_raw_slice",
            "task_specific_gate",
            "rgb_aux_dnn_finetune",
        ],
        "weak_for": ["broad_all_category_claim"],
        "access": "paper points to MindSpore/Gitee RAOD code and dataset path",
        "source_url": "https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Toward_RAW_Object_Detection_A_New_Benchmark_and_a_New_CVPR_2023_paper.html",
        "risk": "Download path/access should be verified before planning automation; Gitee availability can be less convenient than GitHub.",
        "next_action": "Verify RAOD download links and metadata format, then run a small ingest smoke test.",
    },
    {
        "name": "LOD",
        "priority": "P1",
        "recommended_use": "Fast real low-light RAW/RGB paired object-detection slice.",
        "task": "object_detection",
        "domain": "very low light daily scenes",
        "raw_type": "Canon EOS 5D Mark IV RAW-normal/RAW-dark plus RGB-normal/RGB-dark",
        "scale": "2,230 low-light images in RAW-Adapter table; VOC-style XML annotations",
        "conditions": ["low_light", "normal_light_pair"],
        "strong_for": [
            "low_light_slice",
            "humanisp_vs_perceptionisp_raw_pair",
            "task_specific_gate",
        ],
        "weak_for": ["automotive_hdr_claim", "weather_claim"],
        "access": "official GitHub uses Baidu links for RAW/RGB images, annotations, and original raw sensor data",
        "source_url": "https://github.com/ying-fu/LODDataset",
        "risk": "Baidu download can be slow or inconvenient outside China; annotations are XML, so a converter is needed.",
        "next_action": "Try annotation and small RAW-dark/RAW-normal download first, then write VOC-to-internal conversion.",
    },
    {
        "name": "PASCALRAW",
        "priority": "P2",
        "recommended_use": "Controlled daylight sanity check against prior RAW object-detection papers.",
        "task": "object_detection",
        "domain": "daytime street scenes",
        "raw_type": "12-bit Nikon D3200 Bayer RAW with RGB and boxes",
        "scale": "4,259 RAW images; person/car/bicycle; 6,550 object instances",
        "conditions": ["daylight"],
        "strong_for": ["paper_reproduction", "quick_raw_detector_sanity"],
        "weak_for": ["adverse_claim", "small_object_claim", "automotive_hdr_claim"],
        "access": "Stanford Digital Repository PURL is cited by multiple papers",
        "source_url": "http://purl.stanford.edu/hq050zr7488",
        "risk": "Daylight and large-object bias make it weak evidence for our strongest PerceptionISP claims.",
        "next_action": "Use only as a reproducibility baseline after stronger adverse/native choices are queued.",
    },
    {
        "name": "MultiRAW / rho-Vision",
        "priority": "P2",
        "recommended_use": "Sensor-generalization and RAW-domain detection/segmentation supplement.",
        "task": "object_detection_and_segmentation",
        "domain": "multi-camera real RAW snapshots",
        "raw_type": "real RAW from multiple sensors with corresponding RGB and labels",
        "scale": ">7k RAW images; multiple camera sensors; detection and segmentation labels",
        "conditions": ["multi_sensor", "day_night", "driving_like"],
        "strong_for": ["sensor_generalization", "rgb_to_raw_simulation", "segmentation_supplement"],
        "weak_for": ["primary_adverse_weather_claim"],
        "access": "project page and GitHub provide dataset and CycleR2R resources",
        "source_url": "https://github.com/NJUVISION/rho-vision",
        "risk": "Less direct than AODRaw/ROD for our object-detection claim gates; may require adapting dataset layout.",
        "next_action": "Use after AODRaw/ROD/LOD decision, especially if sensor-generalization is the next claim.",
    },
    {
        "name": "ADE20K-RAW",
        "priority": "P3",
        "recommended_use": "Synthetic RAW segmentation supplement only.",
        "task": "semantic_segmentation",
        "domain": "synthetic RAW from ADE20K",
        "raw_type": "synthetic RAW",
        "scale": "27,574 synthetic RAW segmentation images in RAW-Adapter table",
        "conditions": ["synthetic"],
        "strong_for": ["segmentation_supplement", "adapter_smoke_test"],
        "weak_for": ["real_raw_claim", "native_sensor_claim", "object_detection_primary_claim"],
        "access": "RAW-Adapter code/release context",
        "source_url": "https://github.com/cuiziteng/ECCV_RAW_Adapter",
        "risk": "Synthetic-only evidence must not be promoted as real native RAW proof.",
        "next_action": "Use only as a secondary segmentation/adapter experiment.",
    },
)


CLAIM_PRIORITIES: tuple[Dict[str, Any], ...] = (
    {
        "claim": "Adverse-condition/native RAW slice",
        "best_datasets": ["AODRaw", "ROD / RAOD", "LOD"],
        "reason": "These provide real RAW under low-light/HDR/weather-like conditions rather than only simulated sensor noise.",
    },
    {
        "claim": "Automotive HDR/glare",
        "best_datasets": ["ROD / RAOD", "AODRaw"],
        "reason": "ROD is the cleanest driving HDR RAW target; AODRaw adds broader adverse condition diversity.",
    },
    {
        "claim": "RGB+Aux DNN fine-tune gate",
        "best_datasets": ["AODRaw", "ROD / RAOD", "LOD", "MultiRAW / rho-Vision"],
        "reason": "The detector must actually consume RAW-derived/Aux evidence on a held-out labeled split.",
    },
    {
        "claim": "Fast low-light feasibility",
        "best_datasets": ["LOD", "AODRaw"],
        "reason": "LOD is smaller and paired; AODRaw is stronger but heavier.",
    },
    {
        "claim": "Paper-reproduction sanity check",
        "best_datasets": ["PASCALRAW"],
        "reason": "Useful for comparing with prior RAW object-detection baselines, but weak for adverse or automotive claims.",
    },
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create a PerceptionISP RAW dataset readiness report.")
    parser.add_argument("--output-dir", default="reports/perception_raw_dataset_readiness")
    args = parser.parse_args(argv)

    summary = build_raw_dataset_readiness()
    html_path = write_raw_dataset_readiness(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "dataset_count": len(summary["datasets"]),
                    "top_priority": [row["name"] for row in summary["datasets"] if row["priority"] == "P0"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_raw_dataset_readiness() -> Dict[str, Any]:
    datasets = [dict(row) for row in DATASETS]
    return {
        "name": "PerceptionISP RAW dataset readiness",
        "status": "pass",
        "datasets": datasets,
        "claim_priorities": [dict(row) for row in CLAIM_PRIORITIES],
        "recommended_sequence": [
            "Do not bulk-download hundreds of GB before access, license, and storage are checked.",
            "Start with AODRaw annotations plus a small downsampled/sliced RAW subset if storage allows.",
            "In parallel, verify ROD/RAOD availability for automotive HDR/glare evidence.",
            "Use LOD as the fastest real low-light RAW/RGB pair smoke test if AODRaw or ROD access is slow.",
            "Keep PASCALRAW and ADE20K-RAW as sanity/supplementary evidence, not primary claim evidence.",
        ],
        "claim_boundary": (
            "Dataset availability is planning evidence only. Performance claims still require ingest, fixed HumanISP/PerceptionISP protocol, "
            "held-out detector evaluation, and ideally RGB+Aux DNN fine-tuning."
        ),
    }


def write_raw_dataset_readiness(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _render_html(summary: Mapping[str, Any]) -> str:
    dataset_rows = "".join(_dataset_row(row) for row in summary.get("datasets", ()) if isinstance(row, Mapping))
    claim_rows = "".join(_claim_row(row) for row in summary.get("claim_priorities", ()) if isinstance(row, Mapping))
    sequence = "".join(
        f"<li>{html_lib.escape(str(item))}</li>"
        for item in summary.get("recommended_sequence", ())
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PerceptionISP RAW Dataset Readiness</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8faf9; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border-bottom: 1px solid #d8ded7; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #e8f3f1; }}
    .note {{ border-left: 5px solid #2563eb; background: #eff6ff; padding: 12px 14px; margin: 16px 0; }}
    code {{ background: #eef2f4; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PerceptionISP RAW Dataset Readiness</h1>
  <div class="note">{html_lib.escape(str(summary.get('claim_boundary', '')))}</div>
  <h2>Recommended Sequence</h2>
  <ol>{sequence}</ol>
  <h2>Dataset Candidates</h2>
  <table>
    <thead><tr><th>Priority</th><th>Dataset</th><th>Use</th><th>Task/Domain</th><th>Scale</th><th>Strong For</th><th>Risk</th><th>Next Action</th><th>Source</th></tr></thead>
    <tbody>{dataset_rows}</tbody>
  </table>
  <h2>Claim Fit</h2>
  <table>
    <thead><tr><th>Claim</th><th>Best Datasets</th><th>Reason</th></tr></thead>
    <tbody>{claim_rows}</tbody>
  </table>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _dataset_row(row: Mapping[str, Any]) -> str:
    source = str(row.get("source_url", ""))
    source_link = f"<a href=\"{html_lib.escape(source)}\">source</a>" if source else ""
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('priority', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('name', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('recommended_use', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('task', '')))}<br>{html_lib.escape(str(row.get('domain', '')))}<br>{html_lib.escape(str(row.get('raw_type', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('scale', '')))}<br>conditions: {html_lib.escape(', '.join(str(value) for value in row.get('conditions', ())))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('strong_for', ())))}</td>"
        f"<td>{html_lib.escape(str(row.get('risk', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('next_action', '')))}</td>"
        f"<td>{source_link}</td>"
        "</tr>"
    )


def _claim_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('claim', '')))}</td>"
        f"<td>{html_lib.escape(', '.join(str(value) for value in row.get('best_datasets', ())))}</td>"
        f"<td>{html_lib.escape(str(row.get('reason', '')))}</td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
