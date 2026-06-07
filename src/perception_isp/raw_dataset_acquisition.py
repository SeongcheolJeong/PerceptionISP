"""Audit RAW dataset acquisition links before large downloads."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
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
        "priority": "P1",
        "resource": "annotations_terabox",
        "kind": "annotation",
        "url": "https://terabox.com/s/1_shu40gxKZl3XMN99SF2YQ",
        "expected_access": "browser_or_login_possible",
        "disk_estimate_gb": 0.1,
        "first_action": "Use only as an alternate annotation path if Google Drive access fails.",
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
        "priority": "P1",
        "resource": "images_downsampled_srgb_terabox",
        "kind": "image_subset",
        "url": "https://terabox.com/s/1QerBpH6FaGCE05cXks2XxQ",
        "expected_access": "browser_or_login_possible",
        "disk_estimate_gb": 4.3,
        "first_action": "Alternate sRGB baseline path; it does not unblock RAW detection evidence.",
    },
    {
        "dataset": "AODRaw",
        "priority": "P0",
        "resource": "images_downsampled_raw_test_baidu",
        "kind": "raw_images",
        "url": "https://pan.baidu.com/s/1QvqKuBPIgWXzdoABo-L-MQ?pwd=5v4a",
        "expected_access": "baidu",
        "disk_estimate_gb": 58.94,
        "first_action": "Download AODRaw_test_downsampled_raw.zip first, then extract only the needed subset files.",
    },
    {
        "dataset": "AODRaw",
        "priority": "P1",
        "resource": "images_downsampled_raw_train_baidu",
        "kind": "raw_images",
        "url": "https://pan.baidu.com/s/1QvqKuBPIgWXzdoABo-L-MQ?pwd=5v4a",
        "expected_access": "baidu",
        "disk_estimate_gb": 137.19,
        "first_action": "Defer train RAW zip until the test RAW gate passes and disk headroom is confirmed.",
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
        "dataset": "AODRaw",
        "priority": "P2",
        "resource": "images_original_terabox",
        "kind": "raw_images",
        "url": "https://terabox.com/s/1QMnQ7z0V9Wy79pBylG5ZBw",
        "expected_access": "browser_or_login_possible",
        "disk_estimate_gb": 435.0,
        "first_action": "Only consider if downsampled RAW remains blocked and disk/network budget can support original RAW preprocessing.",
    },
    {
        "dataset": "AODRaw",
        "priority": "P2",
        "resource": "images_slice_srgb_terabox",
        "kind": "image_subset",
        "url": "https://terabox.com/s/1FOtFtFbRbmghYsCCodqwUg",
        "expected_access": "browser_or_login_possible",
        "disk_estimate_gb": 23.0,
        "first_action": "Alternate sliced sRGB path for detector baseline only; not a RAW sensor evidence substitute.",
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
        "dataset": "SID",
        "priority": "P1",
        "resource": "official_project",
        "kind": "metadata",
        "url": "https://cchen156.github.io/SID.html",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Use as a low-light RAW/reference diagnostic dataset; it has paired RAW references but no object-detection boxes.",
    },
    {
        "dataset": "SID",
        "priority": "P1",
        "resource": "official_repo",
        "kind": "code",
        "url": "https://github.com/cchen156/Learning-to-See-in-the-Dark",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Use the official file lists to pair short-exposure RAW with long-exposure references.",
    },
    {
        "dataset": "SID",
        "priority": "P1",
        "resource": "sony_raw_google_storage",
        "kind": "raw_images",
        "url": "https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip",
        "expected_access": "public_direct",
        "disk_estimate_gb": 25.08,
        "first_action": "Download Sony2025.zip for real low-light Bayer RAW edge/confidence diagnostics.",
    },
    {
        "dataset": "SID",
        "priority": "P2",
        "resource": "fuji_raw_google_storage",
        "kind": "raw_images",
        "url": "https://storage.googleapis.com/isl-datasets/SID/Fuji2025.zip",
        "expected_access": "public_direct",
        "disk_estimate_gb": 51.61,
        "first_action": "Defer until Sony RAW diagnostics are working; Fuji uses a different sensor pattern.",
    },
    {
        "dataset": "RAW-NOD",
        "priority": "P1",
        "resource": "official_repo",
        "kind": "metadata",
        "url": "https://github.com/igor-morawski/RAW-NOD",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Use as a low-light RAW object-detection candidate after requesting dataset access.",
    },
    {
        "dataset": "RAW-NOD",
        "priority": "P1",
        "resource": "dataset_request_form",
        "kind": "raw_images",
        "url": "https://docs.google.com/forms/d/1aIKTV6026daYFRtje7zcx4LeDz68AOcpWIH7XxNCICY/viewform",
        "expected_access": "google_form_approval",
        "disk_estimate_gb": None,
        "first_action": "Submit the form for access; it is a true low-light RAW object-detection dataset but not immediately downloadable.",
    },
    {
        "dataset": "QUT low-light RAW",
        "priority": "P2",
        "resource": "dataset_page",
        "kind": "metadata",
        "url": "https://open.qcr.ai/dataset/low-light/",
        "expected_access": "public",
        "disk_estimate_gb": 0.0,
        "first_action": "Use only as controlled low-light DNG diagnostics; it is not a detector benchmark.",
    },
    {
        "dataset": "QUT low-light RAW",
        "priority": "P2",
        "resource": "cloudstor_zip",
        "kind": "raw_images",
        "url": "https://cloudstor.aarnet.edu.au/plus/index.php/s/gdJNon8OdEnQeXU/download",
        "expected_access": "public_direct",
        "disk_estimate_gb": 23.3,
        "first_action": "Try only if the CloudStor link resolves; current public page may point at a retired host.",
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
    parser.add_argument("--check-local-state", action="store_true", help="Inspect local dataset/download files before recommending the next action.")
    parser.add_argument("--dataset-root", default="data/raw_datasets", help="Base raw-dataset directory used by --check-local-state.")
    parser.add_argument("--download-root", action="append", default=[], help="Directory to scan for manual downloads. Repeatable; defaults to ~/Downloads.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    summary = build_raw_dataset_acquisition(
        check_network=bool(args.check_network),
        timeout_seconds=float(args.timeout),
        check_local_state=bool(args.check_local_state),
        dataset_root=args.dataset_root,
        download_roots=tuple(args.download_root) if args.download_root else (Path.home() / "Downloads",),
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
                    "local_state_checked": bool(summary["local_state_checked"]),
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
    check_local_state: bool = False,
    dataset_root: str | Path = "data/raw_datasets",
    download_roots: Sequence[str | Path] = (Path.home() / "Downloads",),
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    opener: ResourceOpener | None = None,
) -> Dict[str, Any]:
    open_resource = opener or _open_resource
    local_state = (
        _build_local_state(dataset_root=dataset_root, download_roots=download_roots)
        if check_local_state
        else _empty_local_state(dataset_root=dataset_root, download_roots=download_roots)
    )
    resources = []
    for raw_resource in DATASET_RESOURCES:
        resource = dict(raw_resource)
        if check_network:
            resource["network"] = dict(open_resource(str(resource["url"]), float(timeout_seconds)))
        else:
            resource["network"] = {"checked": False, "status": "not_checked"}
        resource["local_status"] = _resource_local_status(resource, local_state)
        resource["acquisition_status"] = _acquisition_status(resource)
        resource["acquisition_status"] = _apply_local_status(resource["acquisition_status"], resource["local_status"])
        resource["blocker"] = _blocker(resource)
        resources.append(resource)
    dataset_summary = _dataset_summary(resources)
    return {
        "name": "PerceptionISP RAW dataset acquisition audit",
        "status": "pass",
        "network_checked": bool(check_network),
        "local_state_checked": bool(check_local_state),
        "checked_at_epoch": time.time() if check_network else None,
        "local_state": local_state,
        "resources": resources,
        "datasets": dataset_summary,
        "recommended_first": _recommended_first(resources, local_state=local_state if check_local_state else None),
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
    if expected in {"baidu", "browser_or_gdrive", "browser_or_login_possible", "google_form_approval"}:
        return "manual_access_likely"
    if expected == "public_direct":
        return "ready_to_try"
    if str(resource.get("kind", "")) in {"metadata", "code", "annotation"}:
        return "ready_to_try"
    return "needs_manual_check"


def _apply_local_status(acquisition_status: str, local_status: Mapping[str, Any]) -> str:
    status = str(local_status.get("status", "not_checked"))
    if status == "present":
        return "local_available"
    if status in {"partial", "invalid"}:
        return "local_invalid_retry"
    return acquisition_status


def _blocker(resource: Mapping[str, Any]) -> str:
    local = resource.get("local_status", {}) if isinstance(resource.get("local_status"), Mapping) else {}
    local_status = str(local.get("status", ""))
    if local_status in {"partial", "invalid"}:
        return str(local.get("message", "local file is incomplete or invalid; download again"))
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
                "local_available_count": sum(1 for row in dataset_resources if row.get("acquisition_status") == "local_available"),
                "local_invalid_count": sum(1 for row in dataset_resources if row.get("acquisition_status") == "local_invalid_retry"),
            }
        )
    return rows


def _recommended_first(resources: Sequence[Mapping[str, Any]], *, local_state: Mapping[str, Any] | None = None) -> str:
    if local_state is not None:
        if bool(local_state.get("aodraw_annotations_present")):
            test_raw = local_state.get("aodraw_test_raw_zip", {}) if isinstance(local_state.get("aodraw_test_raw_zip"), Mapping) else {}
            srgb = local_state.get("aodraw_srgb_zip", {}) if isinstance(local_state.get("aodraw_srgb_zip"), Mapping) else {}
            if str(test_raw.get("status", "missing")) != "present":
                return "AODRaw:images_downsampled_raw_test_baidu"
            if str(srgb.get("status", "missing")) != "present":
                return "AODRaw:images_downsampled_srgb_baidu"
            return "AODRaw:run_post_download_pipeline"
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


def _empty_local_state(*, dataset_root: str | Path, download_roots: Sequence[str | Path]) -> Dict[str, Any]:
    return {
        "checked": False,
        "dataset_root": str(Path(dataset_root).expanduser()),
        "download_roots": [str(Path(item).expanduser()) for item in download_roots],
    }


def _build_local_state(*, dataset_root: str | Path, download_roots: Sequence[str | Path]) -> Dict[str, Any]:
    root = Path(dataset_root).expanduser()
    aodraw_root = root / "aodraw" if root.name != "aodraw" else root
    search_roots = [Path(item).expanduser() for item in download_roots] + [aodraw_root, aodraw_root / "downloads"]
    annotations = aodraw_root / "annotations" / "AODRaw_annotations.zip"
    test_raw = _find_download_candidate(search_roots, "AODRaw_test_downsampled_raw.zip", expected_bytes=58936290415)
    train_raw = _find_download_candidate(search_roots, "AODRaw_train_downsampled_raw.zip", expected_bytes=137186166557)
    srgb = _find_srgb_candidate(search_roots)
    disk_probe = root if root.exists() else root.parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    usage = shutil.disk_usage(disk_probe)
    return {
        "checked": True,
        "dataset_root": str(root),
        "aodraw_root": str(aodraw_root),
        "download_roots": [str(path) for path in search_roots],
        "disk_available_gib": round(float(usage.free) / float(1024**3), 2),
        "aodraw_annotations_present": bool(annotations.is_file()),
        "aodraw_annotations_path": str(annotations) if annotations.is_file() else "",
        "aodraw_test_raw_zip": test_raw,
        "aodraw_train_raw_zip": train_raw,
        "aodraw_srgb_zip": srgb,
        "sid_sony_zip": _find_download_candidate(search_roots + [root / "sid" / "downloads"], "Sony2025.zip", expected_bytes=26926662016),
        "sid_fuji_zip": _find_download_candidate(search_roots + [root / "sid" / "downloads"], "Fuji2025.zip", expected_bytes=55409370853),
        "qut_low_light_zip": _find_download_candidate(
            search_roots + [root / "qut_low_light" / "downloads"],
            "qut_low_light_raw_dataset.zip",
            expected_bytes=int(23.3 * 1024**3),
        ),
        "next_local_action": _local_next_action(annotations_present=annotations.is_file(), test_raw=test_raw, srgb=srgb),
    }


def _find_srgb_candidate(search_roots: Sequence[Path]) -> Dict[str, Any]:
    names = ("AODRaw_downsampled_srgb.zip", "AODRaw_images_downsampled_srgb.zip", "images_downsampled_srgb.zip")
    candidates = []
    for name in names:
        candidate = _find_download_candidate(search_roots, name, expected_bytes=int(4.3 * 1024**3), validate_zip=True)
        if candidate.get("status") != "missing":
            candidates.append(candidate)
    if not candidates:
        return _missing_candidate(names[0])
    severity = {"present": 0, "partial": 1, "invalid": 2, "missing": 3}
    return sorted(candidates, key=lambda item: severity.get(str(item.get("status")), 9))[0]


def _find_download_candidate(
    search_roots: Sequence[Path],
    filename: str,
    *,
    expected_bytes: int,
    validate_zip: bool = False,
) -> Dict[str, Any]:
    for root in search_roots:
        path = root / filename
        if not path.is_file():
            continue
        size = int(path.stat().st_size)
        if size < int(expected_bytes * 0.95):
            return {
                "filename": filename,
                "path": str(path),
                "status": "partial",
                "size_bytes": size,
                "expected_bytes": int(expected_bytes),
                "message": f"local file is only {round(size / 1024**2, 2)} MiB; expected about {round(expected_bytes / 1024**3, 2)} GiB",
            }
        if validate_zip and not zipfile.is_zipfile(path):
            return {
                "filename": filename,
                "path": str(path),
                "status": "invalid",
                "size_bytes": size,
                "expected_bytes": int(expected_bytes),
                "message": "local file is not a readable zip archive",
            }
        return {
            "filename": filename,
            "path": str(path),
            "status": "present",
            "size_bytes": size,
            "expected_bytes": int(expected_bytes),
            "message": "local file is present and size is plausible",
        }
    return _missing_candidate(filename, expected_bytes=expected_bytes)


def _missing_candidate(filename: str, *, expected_bytes: int | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"filename": filename, "path": "", "status": "missing", "message": "not found locally"}
    if expected_bytes is not None:
        payload["expected_bytes"] = int(expected_bytes)
    return payload


def _resource_local_status(resource: Mapping[str, Any], local_state: Mapping[str, Any]) -> Dict[str, Any]:
    if not bool(local_state.get("checked")):
        return {"checked": False, "status": "not_checked", "message": ""}
    dataset = str(resource.get("dataset", ""))
    name = str(resource.get("resource", ""))
    if dataset != "AODRaw":
        if dataset == "SID":
            key_by_resource = {
                "sony_raw_google_storage": "sid_sony_zip",
                "fuji_raw_google_storage": "sid_fuji_zip",
            }
            key = key_by_resource.get(name)
            if key:
                candidate = dict(local_state.get(key, {}) if isinstance(local_state.get(key), Mapping) else {})
                candidate["checked"] = True
                return candidate
        if dataset == "QUT low-light RAW" and name == "cloudstor_zip":
            candidate = dict(local_state.get("qut_low_light_zip", {}) if isinstance(local_state.get("qut_low_light_zip"), Mapping) else {})
            candidate["checked"] = True
            return candidate
        return {"checked": True, "status": "not_applicable", "message": ""}
    if name == "annotations_google_drive":
        present = bool(local_state.get("aodraw_annotations_present"))
        return {
            "checked": True,
            "status": "present" if present else "missing",
            "path": str(local_state.get("aodraw_annotations_path", "")),
            "message": "annotation archive is already local" if present else "annotation archive is not local",
        }
    key_by_resource = {
        "images_downsampled_raw_test_baidu": "aodraw_test_raw_zip",
        "images_downsampled_raw_train_baidu": "aodraw_train_raw_zip",
        "images_downsampled_srgb_baidu": "aodraw_srgb_zip",
    }
    key = key_by_resource.get(name)
    if key:
        candidate = dict(local_state.get(key, {}) if isinstance(local_state.get(key), Mapping) else {})
        candidate["checked"] = True
        return candidate
    return {"checked": True, "status": "not_applicable", "message": ""}


def _local_next_action(*, annotations_present: bool, test_raw: Mapping[str, Any], srgb: Mapping[str, Any]) -> str:
    if not annotations_present:
        return "Download AODRaw_annotations.zip first."
    if str(test_raw.get("status", "missing")) != "present":
        return "Download AODRaw_test_downsampled_raw.zip first; current local state is not evaluation-ready."
    if str(srgb.get("status", "missing")) != "present":
        return "Download a valid downsampled sRGB zip/directory next for paired checks."
    return "Run the AODRaw post-download pipeline."


def _render_html(summary: Mapping[str, Any]) -> str:
    dataset_rows = "".join(_dataset_row(row) for row in summary.get("datasets", ()) if isinstance(row, Mapping))
    resource_rows = "".join(_resource_row(row) for row in summary.get("resources", ()) if isinstance(row, Mapping))
    local_state = summary.get("local_state", {}) if isinstance(summary.get("local_state"), Mapping) else {}
    local_html = _local_state_html(local_state)
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
  <p>Status: <code>{html_lib.escape(str(summary.get('status', '')))}</code>; network checked: <code>{html_lib.escape(str(summary.get('network_checked', '')))}</code>; local checked: <code>{html_lib.escape(str(summary.get('local_state_checked', '')))}</code>; recommended first: <code>{html_lib.escape(str(summary.get('recommended_first', '')))}</code>.</p>
  {local_html}
  <h2>Dataset Summary</h2>
  <table><thead><tr><th>Dataset</th><th>Priority</th><th>Resources</th><th>Ready</th><th>Manual</th><th>Large</th><th>Blocked</th><th>Local</th><th>Invalid</th></tr></thead><tbody>{dataset_rows}</tbody></table>
  <h2>Resources</h2>
  <table><thead><tr><th>Priority</th><th>Dataset</th><th>Resource</th><th>Kind</th><th>Access</th><th>Disk GB</th><th>Status</th><th>Local</th><th>Network</th><th>Blocker</th><th>First Action</th><th>URL</th></tr></thead><tbody>{resource_rows}</tbody></table>
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
        f"<td>{int(row.get('local_available_count', 0))}</td>"
        f"<td>{int(row.get('local_invalid_count', 0))}</td>"
        "</tr>"
    )


def _resource_row(row: Mapping[str, Any]) -> str:
    network = row.get("network", {}) if isinstance(row.get("network"), Mapping) else {}
    local = row.get("local_status", {}) if isinstance(row.get("local_status"), Mapping) else {}
    url = str(row.get("url", ""))
    disk = row.get("disk_estimate_gb")
    disk_text = "unknown" if disk is None else f"{float(disk):.1f}"
    network_text = str(network.get("status", "not_checked"))
    http_status = network.get("http_status")
    if http_status:
        network_text += f" ({http_status})"
    local_text = str(local.get("status", "not_checked"))
    local_message = str(local.get("message", ""))
    if local_message:
        local_text += f": {local_message}"
    return (
        "<tr>"
        f"<td><code>{html_lib.escape(str(row.get('priority', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('dataset', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('resource', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('kind', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('expected_access', '')))}</td>"
        f"<td>{html_lib.escape(disk_text)}</td>"
        f"<td><code>{html_lib.escape(str(row.get('acquisition_status', '')))}</code></td>"
        f"<td>{html_lib.escape(local_text)}</td>"
        f"<td>{html_lib.escape(network_text)}</td>"
        f"<td>{html_lib.escape(str(row.get('blocker', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('first_action', '')))}</td>"
        f"<td><a href=\"{html_lib.escape(url)}\">link</a></td>"
        "</tr>"
    )


def _local_state_html(local_state: Mapping[str, Any]) -> str:
    if not bool(local_state.get("checked")):
        return ""
    rows = [
        ("AODRaw annotations", "present" if bool(local_state.get("aodraw_annotations_present")) else "missing", str(local_state.get("aodraw_annotations_path", ""))),
        ("AODRaw test RAW zip", _candidate_status(local_state.get("aodraw_test_raw_zip")), _candidate_message(local_state.get("aodraw_test_raw_zip"))),
        ("AODRaw train RAW zip", _candidate_status(local_state.get("aodraw_train_raw_zip")), _candidate_message(local_state.get("aodraw_train_raw_zip"))),
        ("AODRaw sRGB zip", _candidate_status(local_state.get("aodraw_srgb_zip")), _candidate_message(local_state.get("aodraw_srgb_zip"))),
        ("SID Sony RAW zip", _candidate_status(local_state.get("sid_sony_zip")), _candidate_message(local_state.get("sid_sony_zip"))),
        ("SID Fuji RAW zip", _candidate_status(local_state.get("sid_fuji_zip")), _candidate_message(local_state.get("sid_fuji_zip"))),
        ("QUT low-light RAW zip", _candidate_status(local_state.get("qut_low_light_zip")), _candidate_message(local_state.get("qut_low_light_zip"))),
    ]
    body = "".join(
        "<tr>"
        f"<td>{html_lib.escape(label)}</td>"
        f"<td><code>{html_lib.escape(status)}</code></td>"
        f"<td>{html_lib.escape(message)}</td>"
        "</tr>"
        for label, status, message in rows
    )
    return (
        "<h2>Local State</h2>"
        f"<p>Next local action: <b>{html_lib.escape(str(local_state.get('next_local_action', '')))}</b> "
        f"Disk available: <code>{html_lib.escape(str(local_state.get('disk_available_gib', '')))} GiB</code>.</p>"
        f"<table><thead><tr><th>Item</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _candidate_status(value: Any) -> str:
    return str(value.get("status", "")) if isinstance(value, Mapping) else ""


def _candidate_message(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    return str(value.get("path") or value.get("message") or "")


if __name__ == "__main__":
    raise SystemExit(main())
