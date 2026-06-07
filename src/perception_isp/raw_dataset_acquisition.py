"""Audit RAW dataset acquisition links before large downloads."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from .types import json_ready


SUMMARY_FILENAME = "raw_dataset_acquisition_summary.json"
DEFAULT_TIMEOUT_SECONDS = 15.0


ResourceOpener = Callable[[str, float], Mapping[str, Any]]


DATASET_RESOURCES: tuple[Dict[str, Any], ...] = (
    {
        "dataset": "AODRaw",
        "priority": "P0",
        "resource": "official_repo",
        "kind": "metadata",
        "url": "https://github.com/lzyhha/AODRaw",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Read README and license.",
    },
    {
        "dataset": "AODRaw",
        "priority": "P0",
        "resource": "training_code",
        "kind": "code",
        "url": "https://github.com/lzyhha/AODRaw-mmdetection",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Inspect configs and annotation loader before downloading images.",
    },
    {
        "dataset": "AODRaw",
        "priority": "P0",
        "resource": "annotations_google_drive",
        "kind": "annotation",
        "url": "https://drive.google.com/file/d/1VEm1TRur7UgjzzEB2vx1kApC1xvceMzS/view?usp=sharing",
        "expected_access": "browser_or_gdrive",
        "disk_estimate_gb": 0.1,
        "first_action": "Try annotation-only download before images.",
    },
    {
        "dataset": "AODRaw",
        "priority": "P0",
        "resource": "images_downsampled_srgb_baidu",
        "kind": "image_subset",
        "url": "https://pan.baidu.com/s/1_56k-Tr1JGDI99xFugPGtQ?pwd=aerr",
        "expected_access": "baidu",
        "disk_estimate_gb": 4.3,
        "first_action": "Use only if annotation ingest works and sRGB baseline is needed.",
    },
    {
        "dataset": "AODRaw",
        "priority": "P0",
        "resource": "images_downsampled_raw_baidu",
        "kind": "raw_images",
        "url": "https://pan.baidu.com/s/1QvqKuBPIgWXzdoABo-L-MQ?pwd=5v4a",
        "expected_access": "baidu",
        "disk_estimate_gb": 223.0,
        "first_action": "Do not download until disk budget and small subset path are confirmed.",
    },
    {
        "dataset": "AODRaw",
        "priority": "P0",
        "resource": "images_original_baidu",
        "kind": "raw_images",
        "url": "https://pan.baidu.com/s/1WqPZz_E9godci3FHlx07EQ?pwd=i2dv",
        "expected_access": "baidu",
        "disk_estimate_gb": 435.0,
        "first_action": "Avoid bulk download for first feasibility run.",
    },
    {
        "dataset": "ROD / RAOD",
        "priority": "P0",
        "resource": "mindspore_raod_readme",
        "kind": "metadata",
        "url": "https://gitee.com/mindspore/models/tree/master/research/cv/RAOD",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Read dataset structure and loader assumptions.",
    },
    {
        "dataset": "ROD / RAOD",
        "priority": "P0",
        "resource": "openi_dataset",
        "kind": "dataset_page",
        "url": "https://openi.pcl.ac.cn/innovation_contest/innov202305091731448/datasets?lang=en-US",
        "expected_access": "browser_or_login_possible",
        "disk_estimate_gb": None,
        "first_action": "Verify whether login or manual acceptance is required.",
    },
    {
        "dataset": "LOD",
        "priority": "P1",
        "resource": "official_repo",
        "kind": "metadata",
        "url": "https://github.com/ying-fu/LODDataset",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Read README and annotation naming rules.",
    },
    {
        "dataset": "LOD",
        "priority": "P1",
        "resource": "raw_dark_annotations_baidu",
        "kind": "annotation",
        "url": "https://pan.baidu.com/s/1pFAwtaX4ufuZaMy31Sv0AA",
        "expected_access": "baidu",
        "disk_estimate_gb": 0.05,
        "first_action": "Try annotation download first, then VOC XML conversion.",
    },
    {
        "dataset": "LOD",
        "priority": "P1",
        "resource": "raw_dark_images_baidu",
        "kind": "raw_images",
        "url": "https://pan.baidu.com/s/1cWu7Y6GtiRV9itZEbop4VQ",
        "expected_access": "baidu",
        "disk_estimate_gb": None,
        "first_action": "Download a small manual subset before full RAW-dark images.",
    },
    {
        "dataset": "PASCALRAW",
        "priority": "P2",
        "resource": "stanford_purl",
        "kind": "dataset_page",
        "url": "http://purl.stanford.edu/hq050zr7488",
        "expected_access": "public",
        "disk_estimate_gb": None,
        "first_action": "Use as reproduction baseline only after adverse datasets are queued.",
    },
    {
        "dataset": "MultiRAW / rho-Vision",
        "priority": "P2",
        "resource": "official_repo",
        "kind": "metadata",
        "url": "https://github.com/NJUVISION/rho-vision",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Use for sensor-generalization after AODRaw/ROD/LOD path is clear.",
    },
    {
        "dataset": "MultiRAW / rho-Vision",
        "priority": "P2",
        "resource": "multiraw_box",
        "kind": "dataset_page",
        "url": "https://box.nju.edu.cn/d/0f4b5206cf734bd889aa/",
        "expected_access": "public_or_institutional_box",
        "disk_estimate_gb": None,
        "first_action": "Check browser access if sensor-generalization becomes the next claim.",
    },
)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Audit RAW dataset acquisition links and disk risk.")
    parser.add_argument("--output-dir", default="reports/perception_raw_dataset_acquisition")
    parser.add_argument("--check-network", action="store_true", help="Issue lightweight HTTP HEAD/GET checks.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    summary = build_raw_dataset_acquisition(
        check_network=bool(args.check_network),
        timeout_seconds=float(args.timeout),
    )
    html_path = write_raw_dataset_acquisition(summary, args.output_dir)
    print(
        json.dumps(
            json_ready(
                {
                    "report": str(html_path),
                    "summary_json": str(html_path.parent / SUMMARY_FILENAME),
                    "status": summary["status"],
                    "resource_count": len(summary["resources"]),
                    "network_checked": bool(summary["network_checked"]),
                    "recommended_first": summary["recommended_first"],
                }
            ),
            indent=2,
        )
    )
    return 0


def build_raw_dataset_acquisition(
    *,
    check_network: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    opener: ResourceOpener | None = None,
) -> Dict[str, Any]:
    open_resource = opener or _open_resource
    resources = []
    for raw_resource in DATASET_RESOURCES:
        resource = dict(raw_resource)
        if check_network:
            resource["network"] = dict(open_resource(str(resource["url"]), float(timeout_seconds)))
        else:
            resource["network"] = {"checked": False, "status": "not_checked"}
        resource["acquisition_status"] = _acquisition_status(resource)
        resource["blocker"] = _blocker(resource)
        resources.append(resource)
    dataset_summary = _dataset_summary(resources)
    return {
        "name": "PerceptionISP RAW dataset acquisition audit",
        "status": "pass",
        "network_checked": bool(check_network),
        "checked_at_epoch": time.time() if check_network else None,
        "resources": resources,
        "datasets": dataset_summary,
        "recommended_first": _recommended_first(resources),
        "download_guardrail": (
            "Do not bulk-download original RAW directories until annotation access, license/access requirements, and disk budget are confirmed. "
            "Prefer annotation-only or small manually selected subsets first."
        ),
        "interpretation": (
            "This report is an acquisition readiness check. It does not prove dataset ingest or detector performance."
        ),
    }


def write_raw_dataset_acquisition(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SUMMARY_FILENAME).write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(summary))
    return html_path


def _open_resource(url: str, timeout_seconds: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "PerceptionISP-raw-dataset-acquisition/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            return {
                "checked": True,
                "status": "reachable",
                "http_status": int(response.status),
                "final_url": response.geturl(),
                "content_type": response.headers.get("content-type", ""),
            }
    except urllib.error.HTTPError as exc:
        if int(exc.code) in {403, 405}:
            return _open_resource_get(url, timeout_seconds)
        return {
            "checked": True,
            "status": "http_error",
            "http_status": int(exc.code),
            "final_url": url,
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - exercised through fake opener in tests.
        return {
            "checked": True,
            "status": "error",
            "final_url": url,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _open_resource_get(url: str, timeout_seconds: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "PerceptionISP-raw-dataset-acquisition/1.0", "Range": "bytes=0-1023"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            return {
                "checked": True,
                "status": "reachable",
                "http_status": int(response.status),
                "final_url": response.geturl(),
                "content_type": response.headers.get("content-type", ""),
            }
    except Exception as exc:
        return {
            "checked": True,
            "status": "error",
            "final_url": url,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _acquisition_status(resource: Mapping[str, Any]) -> str:
    network = resource.get("network", {}) if isinstance(resource.get("network"), Mapping) else {}
    if bool(network.get("checked")) and str(network.get("status")) not in {"reachable", "not_checked"}:
        return "blocked_by_network"
    expected = str(resource.get("expected_access", ""))
    disk = resource.get("disk_estimate_gb")
    if disk is not None and float(disk) >= 100.0:
        return "defer_large_download"
    if expected in {"baidu", "browser_or_gdrive", "browser_or_login_possible"}:
        return "manual_access_likely"
    if str(resource.get("kind", "")) in {"metadata", "code", "annotation"}:
        return "ready_to_try"
    return "needs_manual_check"


def _blocker(resource: Mapping[str, Any]) -> str:
    status = str(resource.get("acquisition_status", ""))
    if status == "blocked_by_network":
        network = resource.get("network", {}) if isinstance(resource.get("network"), Mapping) else {}
        return str(network.get("error") or network.get("status") or "network check failed")
    if status == "defer_large_download":
        return f"estimated {resource.get('disk_estimate_gb')} GB; start with annotations or subset"
    if status == "manual_access_likely":
        return f"expected access path is {resource.get('expected_access')}"
    return ""


def _dataset_summary(resources: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    datasets = sorted({str(row.get("dataset", "")) for row in resources})
    rows = []
    for dataset in datasets:
        dataset_resources = [row for row in resources if str(row.get("dataset", "")) == dataset]
        rows.append(
            {
                "dataset": dataset,
                "priority": min(str(row.get("priority", "P9")) for row in dataset_resources),
                "resource_count": len(dataset_resources),
                "ready_to_try_count": sum(1 for row in dataset_resources if row.get("acquisition_status") == "ready_to_try"),
                "manual_access_count": sum(1 for row in dataset_resources if row.get("acquisition_status") == "manual_access_likely"),
                "large_download_count": sum(1 for row in dataset_resources if row.get("acquisition_status") == "defer_large_download"),
                "blocked_count": sum(1 for row in dataset_resources if row.get("acquisition_status") == "blocked_by_network"),
            }
        )
    return rows


def _recommended_first(resources: Sequence[Mapping[str, Any]]) -> str:
    for dataset, resource in (
        ("AODRaw", "annotations_google_drive"),
        ("ROD / RAOD", "openi_dataset"),
        ("LOD", "raw_dark_annotations_baidu"),
    ):
        row = next(
            (
                item
                for item in resources
                if str(item.get("dataset", "")) == dataset and str(item.get("resource", "")) == resource
            ),
            None,
        )
        if row is None:
            continue
        if str(row.get("acquisition_status", "")) != "blocked_by_network":
            return f"{dataset}:{resource}"
    return "none"


def _render_html(summary: Mapping[str, Any]) -> str:
    dataset_rows = "".join(_dataset_row(row) for row in summary.get("datasets", ()) if isinstance(row, Mapping))
    resource_rows = "".join(_resource_row(row) for row in summary.get("resources", ()) if isinstance(row, Mapping))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PerceptionISP RAW Dataset Acquisition Audit</title>
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
  <h1>PerceptionISP RAW Dataset Acquisition Audit</h1>
  <div class="note">{html_lib.escape(str(summary.get('download_guardrail', '')))}</div>
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; network checked: <code>{html_lib.escape(str(summary.get('network_checked', '')))}</code>; recommended first: <code>{html_lib.escape(str(summary.get('recommended_first', '')))}</code>.</p>
  <h2>Dataset Summary</h2>
  <table><thead><tr><th>Dataset</th><th>Priority</th><th>Resources</th><th>Ready</th><th>Manual</th><th>Large</th><th>Blocked</th></tr></thead><tbody>{dataset_rows}</tbody></table>
  <h2>Resources</h2>
  <table><thead><tr><th>Priority</th><th>Dataset</th><th>Resource</th><th>Kind</th><th>Access</th><th>Disk GB</th><th>Status</th><th>Network</th><th>Blocker</th><th>First Action</th><th>URL</th></tr></thead><tbody>{resource_rows}</tbody></table>
  <p>{html_lib.escape(str(summary.get('interpretation', '')))}</p>
  <p>Raw JSON: <code>{SUMMARY_FILENAME}</code></p>
</body>
</html>
"""


def _dataset_row(row: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('dataset', '')))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('priority', '')))}</code></td>"
        f"<td>{int(row.get('resource_count', 0))}</td>"
        f"<td>{int(row.get('ready_to_try_count', 0))}</td>"
        f"<td>{int(row.get('manual_access_count', 0))}</td>"
        f"<td>{int(row.get('large_download_count', 0))}</td>"
        f"<td>{int(row.get('blocked_count', 0))}</td>"
        "</tr>"
    )


def _resource_row(row: Mapping[str, Any]) -> str:
    network = row.get("network", {}) if isinstance(row.get("network"), Mapping) else {}
    url = str(row.get("url", ""))
    disk = row.get("disk_estimate_gb")
    disk_text = "unknown" if disk is None else f"{float(disk):.1f}"
    network_text = str(network.get("status", "not_checked"))
    http_status = network.get("http_status")
    if http_status:
        network_text += f" ({http_status})"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('priority', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('dataset', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('resource', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('kind', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('expected_access', '')))}</td>"
        f"<td>{html_lib.escape(disk_text)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('acquisition_status', '')))}</code></td>"
        f"<td>{html_lib.escape(network_text)}</td>"
        f"<td>{html_lib.escape(str(row.get('blocker', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('first_action', '')))}</td>"
        f"<td><a href=\"{html_lib.escape(url)}\">link</a></td>"
        "</tr>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
