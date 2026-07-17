"""Build a deterministic, path-scrubbed bundle of public HTML reports.

The full nuScenes showcase is an acceptance artifact: it includes large NPY
arrays and machine-local reproduction paths.  This module creates a separate
publication bundle containing the human-readable reports and PNG previews
without mutating that acceptance artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Final

from PIL import Image


PUBLIC_BUNDLE_SCHEMA_VERSION: Final = "perception_isp_public_report_bundle_v1"
SOURCE_RGB_PREVIEW_MAX_EDGE_PX: Final = 800

SHOWCASE_SOURCE_FILES: Final = (
    "EVIDENCE.md",
    "aux_map_catalog.json",
    "environment.json",
    "exposure_plane_sources.csv",
    "index.html",
    "metadata_origins.csv",
    "source_manifest.json",
    "summary.json",
)

_STATUS_SOURCE_FILES: Final = ("index.html", "project_status_summary.json")
_STALE_EVIDENCE_FILES: Final = frozenset(
    {"artifacts_manifest.json", "checksums.sha256"}
)
_TEXT_SUFFIXES: Final = frozenset(
    {".css", ".csv", ".html", ".js", ".json", ".md", ".sha256", ".svg", ".txt", ".xml"}
)
_EXTRA_REPORT_SUFFIXES: Final = _TEXT_SUFFIXES | {".png"}
_FORBIDDEN_LOCAL_PREFIXES: Final = ("/Users/", "/tmp/")


@dataclass(frozen=True)
class PublicReportBundle:
    """Paths to the reports written below a public bundle root."""

    root: Path
    status_report: Path
    showcase_report: Path
    extra_reports: tuple[Path, ...]


def build_public_report_bundle(
    *,
    status_report_dir: str | os.PathLike[str],
    showcase_report_dir: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    local_prefix_redactions: Mapping[str | os.PathLike[str], str] | None = None,
    extra_report_dirs: Sequence[str | os.PathLike[str]] = (),
) -> PublicReportBundle:
    """Create a lightweight, deterministic copy suitable for publication.

    ``destination`` must not exist.  Each source report keeps its directory
    name so relative links between reports remain valid.  Text is redacted by
    literal prefix replacement, longest prefix first.  Any remaining ``/Users/``
    or ``/tmp/`` reference rejects the whole build.

    The showcase copy contains exactly the eight top-level source documents,
    all ``assets/**/*.png`` previews, a data notice, and newly computed public
    artifact/checksum manifests.  It intentionally excludes ``arrays/**`` and
    the source bundle's stale manifests.
    """

    status_source = _required_directory(status_report_dir, label="status report")
    showcase_source = _required_directory(showcase_report_dir, label="showcase report")
    extra_sources = tuple(
        _required_directory(value, label="extra report") for value in extra_report_dirs
    )
    sources = (status_source, showcase_source, *extra_sources)
    _require_distinct_report_names(sources)

    output = Path(destination).expanduser()
    if output.exists():
        raise FileExistsError(f"public report bundle destination already exists: {output}")
    _reject_destination_inside_sources(output, sources)

    redactions = _validated_redactions(local_prefix_redactions or {})
    _require_files(status_source, _STATUS_SOURCE_FILES, label="status report")
    _require_files(showcase_source, SHOWCASE_SOURCE_FILES, label="showcase report")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=str(output.parent))
    )
    try:
        status_output = staging / status_source.name
        showcase_output = staging / showcase_source.name
        _copy_named_text_files(status_source, status_output, _STATUS_SOURCE_FILES, redactions)
        _copy_showcase(showcase_source, showcase_output, redactions)

        extra_outputs = []
        for source in extra_sources:
            extra_output = staging / source.name
            _copy_extra_report(source, extra_output, redactions)
            extra_outputs.append(extra_output)

        _reject_residual_local_paths(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return PublicReportBundle(
        root=output,
        status_report=output / status_source.name / "index.html",
        showcase_report=output / showcase_source.name / "index.html",
        extra_reports=tuple(output / source.name / "index.html" for source in extra_sources),
    )


def _required_directory(value: str | os.PathLike[str], *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} directory does not exist: {path}")
    return path


def _require_distinct_report_names(sources: Sequence[Path]) -> None:
    names = [source.name for source in sources]
    if any(not name or name in {".", ".."} for name in names):
        raise ValueError("source report directories must have safe, non-empty names")
    if len(set(names)) != len(names):
        raise ValueError(f"source report directory names must be distinct: {names}")


def _reject_destination_inside_sources(destination: Path, sources: Sequence[Path]) -> None:
    resolved_output = destination.resolve()
    for source in sources:
        resolved_source = source.resolve()
        if resolved_output == resolved_source or resolved_source in resolved_output.parents:
            raise ValueError(
                f"public bundle destination must not be inside a source report: {destination}"
            )


def _validated_redactions(
    candidates: Mapping[str | os.PathLike[str], str],
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for raw_prefix, raw_replacement in candidates.items():
        prefix = os.fspath(raw_prefix)
        replacement = str(raw_replacement)
        if not prefix:
            raise ValueError("redaction prefix must not be empty")
        if not replacement:
            raise ValueError(f"redaction replacement must not be empty for {prefix!r}")
        if any(value in replacement for value in _FORBIDDEN_LOCAL_PREFIXES):
            raise ValueError(
                f"redaction replacement still contains a local path prefix: {replacement!r}"
            )
        rows.append((prefix, replacement))
    return tuple(sorted(rows, key=lambda row: (-len(row[0]), row[0], row[1])))


def _require_files(source: Path, names: Sequence[str], *, label: str) -> None:
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is missing required files: {missing}")


def _copy_named_text_files(
    source: Path,
    destination: Path,
    names: Sequence[str],
    redactions: Sequence[tuple[str, str]],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        text = (source / name).read_text(encoding="utf-8")
        (destination / name).write_text(
            _redact_text(text, redactions), encoding="utf-8"
        )


def _copy_showcase(
    source: Path,
    destination: Path,
    redactions: Sequence[tuple[str, str]],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    png_sources = _showcase_png_sources(source)
    source_rgb_previews = tuple(
        path for path in png_sources if _is_source_rgb_preview(path)
    )
    source_rgb_preview_path_count = len(source_rgb_previews)
    source_rgb_preview_unique_frame_count = len(
        {_sha256_file(path) for path in source_rgb_previews}
    )
    newly_downscaled_source_rgb_count = 0
    for png_source in png_sources:
        relative = png_source.relative_to(source)
        png_destination = destination / relative
        png_destination.parent.mkdir(parents=True, exist_ok=True)
        newly_downscaled_source_rgb_count += int(
            _copy_public_png_preview(png_source, png_destination)
        )
    downscaled_source_rgb_count = max(
        newly_downscaled_source_rgb_count,
        _prior_downscaled_source_rgb_count(source / "summary.json"),
    )

    for name in SHOWCASE_SOURCE_FILES:
        text = _redact_text((source / name).read_text(encoding="utf-8"), redactions)
        if name == "index.html":
            text = _public_showcase_html(
                text,
                source_rgb_preview_path_count=source_rgb_preview_path_count,
                source_rgb_preview_unique_frame_count=source_rgb_preview_unique_frame_count,
            )
        elif name == "EVIDENCE.md":
            text = _public_evidence_markdown(
                text,
                source_rgb_preview_path_count=source_rgb_preview_path_count,
                source_rgb_preview_unique_frame_count=source_rgb_preview_unique_frame_count,
            )
        elif name == "summary.json":
            text = _public_summary_json(
                text,
                downscaled_source_rgb_count=downscaled_source_rgb_count,
                source_rgb_preview_path_count=source_rgb_preview_path_count,
                source_rgb_preview_unique_frame_count=source_rgb_preview_unique_frame_count,
            )
        (destination / name).write_text(text, encoding="utf-8")

    (destination / "DATA_NOTICE.md").write_text(
        _data_notice(
            report_name=source.name,
            png_count=len(png_sources),
            downscaled_source_rgb_count=downscaled_source_rgb_count,
            source_rgb_preview_path_count=source_rgb_preview_path_count,
            source_rgb_preview_unique_frame_count=source_rgb_preview_unique_frame_count,
        ),
        encoding="utf-8",
    )
    _write_artifacts_manifest(
        destination,
        downscaled_source_rgb_count=downscaled_source_rgb_count,
        source_rgb_preview_path_count=source_rgb_preview_path_count,
        source_rgb_preview_unique_frame_count=source_rgb_preview_unique_frame_count,
    )
    _write_checksums(destination)


def _showcase_png_sources(source: Path) -> tuple[Path, ...]:
    assets = source / "assets"
    if not assets.is_dir():
        raise NotADirectoryError(f"showcase report is missing assets directory: {assets}")
    return tuple(
        sorted(
            (
                path
                for path in assets.rglob("*")
                if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".png"
            ),
            key=lambda path: path.relative_to(source).as_posix(),
        )
    )


def _copy_public_png_preview(source: Path, destination: Path) -> bool:
    """Copy a PNG, bounding only nuScenes source-RGB thumbnails for publication."""

    if not _is_source_rgb_preview(source):
        shutil.copyfile(source, destination)
        return False

    with Image.open(source) as image:
        width, height = image.size
        largest_edge = max(width, height)
        if largest_edge <= SOURCE_RGB_PREVIEW_MAX_EDGE_PX:
            shutil.copyfile(source, destination)
            return False
        scale = SOURCE_RGB_PREVIEW_MAX_EDGE_PX / largest_edge
        output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image.resize(output_size, Image.Resampling.LANCZOS)
        resized.save(destination, format="PNG", compress_level=9, optimize=False)
    return True


def _is_source_rgb_preview(path: Path) -> bool:
    return path.name.lower().endswith("_source_jpeg_rgb.png")


def _prior_downscaled_source_rgb_count(summary_path: Path) -> int:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return 0
    value = payload.get("source_rgb_preview_downscaled_count", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "source_rgb_preview_downscaled_count must be a non-negative integer"
        )
    return value


def _copy_extra_report(
    source: Path,
    destination: Path,
    redactions: Sequence[tuple[str, str]],
) -> None:
    """Copy a small linked report while applying the same public-data policy."""

    candidates = []
    for path in source.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source)
        if "arrays" in relative.parts or path.name in _STALE_EVIDENCE_FILES:
            continue
        if path.suffix.lower() not in _EXTRA_REPORT_SUFFIXES:
            continue
        candidates.append(path)
    if not any(path.relative_to(source).as_posix() == "index.html" for path in candidates):
        raise FileNotFoundError(f"extra report is missing index.html: {source}")

    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(candidates, key=lambda value: value.relative_to(source).as_posix()):
        relative = path.relative_to(source)
        copied = destination / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        if _is_text_candidate(path):
            copied.write_text(
                _redact_text(path.read_text(encoding="utf-8"), redactions),
                encoding="utf-8",
            )
        else:
            shutil.copyfile(path, copied)


def _redact_text(text: str, redactions: Sequence[tuple[str, str]]) -> str:
    result = text
    for prefix, replacement in redactions:
        result = result.replace(prefix, replacement)
    return result


def _public_showcase_html(
    text: str,
    *,
    source_rgb_preview_path_count: int,
    source_rgb_preview_unique_frame_count: int,
) -> str:
    public_scope = (
        "<p class='public-bundle-notice'><b>Public preview bundle:</b> PNG previews and "
        "textual provenance are included; lossless numerical arrays are intentionally "
        f"omitted. Source-RGB input thumbnails are bounded to "
        f"{SOURCE_RGB_PREVIEW_MAX_EDGE_PX}px ({source_rgb_preview_path_count} published paths; "
        f"unique frame images: {source_rgb_preview_unique_frame_count}); Aux-map, pseudo-RAW, "
        "and result PNGs are copied byte-for-byte. See "
        "<a href='DATA_NOTICE.md'>DATA_NOTICE.md</a> for "
        "nuScenes-derived preview terms and attribution.</p>"
    )
    paragraph_pattern = re.compile(
        r"<p>(?:(?!</p>).)*(?:<code>arrays/</code>|lossless\s+(?:array|배열))"
        r"(?:(?!</p>).)*</p>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    notice_pattern = re.compile(
        r"<p\s+class=(?:'|\")public-bundle-notice(?:'|\")>(?:(?!</p>).)*</p>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    rendered, replacements = notice_pattern.subn(public_scope, text)
    if replacements == 0:
        rendered, replacements = paragraph_pattern.subn(public_scope, rendered)
    rendered = re.sub(
        r"lossless\s+trace",
        "public-preview textual provenance",
        rendered,
        flags=re.IGNORECASE,
    )
    has_notice_link = (
        "href='DATA_NOTICE.md'" in rendered or 'href="DATA_NOTICE.md"' in rendered
    )
    if replacements == 0 and not has_notice_link:
        rendered = _insert_before_body_end(rendered, public_scope)

    if "href='DATA_NOTICE.md'" not in rendered and 'href="DATA_NOTICE.md"' not in rendered:
        rendered = _insert_before_body_end(rendered, public_scope)
    return rendered


def _public_summary_json(
    text: str,
    *,
    downscaled_source_rgb_count: int,
    source_rgb_preview_path_count: int,
    source_rgb_preview_unique_frame_count: int,
) -> str:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("showcase summary.json must contain a JSON object")
    payload.update(
        {
            "arrays_included": False,
            "data_notice": "DATA_NOTICE.md",
            "interpretation": (
                "The displayed PNG previews and textual provenance in this GitHub preview were "
                "regenerated from the selected nuScenes JPEGs through CameraE2E sensor.volts "
                "pseudo-RAW and PerceptionISP. Lossless numerical arrays from the full local "
                "acceptance bundle are intentionally omitted; any retained array hashes or "
                "statistics document generation lineage but are not downloadable numerical "
                "evidence in this preview."
            ),
            "preview_only": True,
            "publication_profile": "github_preview",
            "source_rgb_preview_downscaled_count": downscaled_source_rgb_count,
            "source_rgb_preview_max_edge_px": SOURCE_RGB_PREVIEW_MAX_EDGE_PX,
            "source_rgb_preview_path_count": source_rgb_preview_path_count,
            "source_rgb_preview_unique_frame_count": source_rgb_preview_unique_frame_count,
        }
    )
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _public_evidence_markdown(
    text: str,
    *,
    source_rgb_preview_path_count: int,
    source_rgb_preview_unique_frame_count: int,
) -> str:
    rows = []
    for line in text.splitlines():
        lowered = line.lower()
        if "`arrays/`" in lowered:
            rows.append(
                "- Lossless numerical arrays are intentionally omitted from this public preview; "
                "use the separately retained local acceptance bundle for numerical-array review."
            )
        elif "png files are visual previews" in lowered and "npy" in lowered:
            rows.append(
                "- PNG files are visualization-only previews; this public bundle does not provide "
                "lossless numerical evidence for quantitative re-analysis."
            )
        else:
            # Generated evidence uses Markdown's two-space line break. Preserve
            # the rendering without committing whitespace-only line endings.
            rows.append(line[:-2] + "<br>" if line.endswith("  ") else line)

    rendered = "\n".join(rows).rstrip() + "\n"
    notice = (
        "\n## Public preview scope\n\n"
        "Lossless numerical arrays from the full local acceptance bundle are not included here. "
        "See [DATA_NOTICE.md](DATA_NOTICE.md) for attribution, license, nuScenes terms, citation, "
        "and non-endorsement language for the PNG previews.\n"
    )
    if "[DATA_NOTICE.md](DATA_NOTICE.md)" not in rendered:
        rendered += notice
    downscale_note = (
        f"\nThe {source_rgb_preview_path_count} published source-RGB paths have a unique "
        f"frame-image count of {source_rgb_preview_unique_frame_count}. Source-RGB input "
        f"thumbnails larger than {SOURCE_RGB_PREVIEW_MAX_EDGE_PX}px on "
        "their longest edge are downscaled for publication; Aux-map, pseudo-RAW, and "
        "PerceptionISP result PNGs are retained byte-for-byte.\n"
    )
    prior_downscale_note = re.compile(
        r"\n(?:The \d+ published source-RGB paths (?:represent \d+ unique frame images|"
        r"have a unique frame-image count of \d+)\. )?"
        r"Source-RGB input thumbnails larger than.*?retained byte-for-byte\.\n"
    )
    rendered, replacements = prior_downscale_note.subn(downscale_note, rendered)
    if replacements == 0:
        rendered += downscale_note
    return rendered


def _insert_before_body_end(html: str, fragment: str) -> str:
    position = html.lower().rfind("</body>")
    if position < 0:
        return html.rstrip() + fragment + "\n"
    return html[:position] + fragment + html[position:]


def _data_notice(
    *,
    report_name: str,
    png_count: int,
    downscaled_source_rgb_count: int,
    source_rgb_preview_path_count: int,
    source_rgb_preview_unique_frame_count: int,
) -> str:
    return f"""# Data and attribution notice

Report: `{report_name}`

Published PNG previews: **{png_count}**

Downscaled source-RGB input previews: **{downscaled_source_rgb_count}**

Source-RGB preview paths: **{source_rgb_preview_path_count}**<br>
Unique source-frame images represented by those paths: **{source_rgb_preview_unique_frame_count}**

Only source-RGB input thumbnails whose largest edge exceeded
{SOURCE_RGB_PREVIEW_MAX_EDGE_PX} pixels were downscaled to that bound for the GitHub preview.
Aux-map, pseudo-RAW, and PerceptionISP result PNGs were copied byte-for-byte. The separately
retained local acceptance bundle and its numerical arrays were not modified.

The PNG visual previews in this report are derived from nuScenes camera JPEGs through the
CameraE2E forward recapture simulator and the PerceptionISP pipeline. They are not native
sensor RAW, inverse-ISP recovery, or a same-instant native HDR bracket.

The nuScenes-derived PNG previews are shared under the
[CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
and remain subject to the applicable [nuScenes terms of use](https://www.nuscenes.org/terms-of-use).
The source nuScenes JPEGs and tables are not redistributed in this bundle. This notice does
not grant rights beyond the nuScenes terms; downstream users must independently confirm that
their use and redistribution are permitted.

Please cite the source dataset paper:

> Holger Caesar et al., “nuScenes: A multimodal dataset for autonomous driving,” CVPR 2020.
> [arXiv:1903.11027](https://arxiv.org/abs/1903.11027)

The nuScenes authors, dataset providers, Motional, and Aptiv do not endorse this report,
PerceptionISP, or the interpretations presented here.
"""


def _write_artifacts_manifest(
    destination: Path,
    *,
    downscaled_source_rgb_count: int,
    source_rgb_preview_path_count: int,
    source_rgb_preview_unique_frame_count: int,
) -> None:
    artifacts = []
    for path in _sorted_files(destination):
        relative = path.relative_to(destination).as_posix()
        if relative in _STALE_EVIDENCE_FILES:
            continue
        artifacts.append(
            {
                "kind": _artifact_kind(relative),
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "notes": {
            "arrays_included": False,
            "checksums_file_included": False,
            "manifest_self_included": False,
            "preview_only": True,
            "source_rgb_preview_downscaled_count": downscaled_source_rgb_count,
            "source_rgb_preview_max_edge_px": SOURCE_RGB_PREVIEW_MAX_EDGE_PX,
            "source_rgb_preview_path_count": source_rgb_preview_path_count,
            "source_rgb_preview_unique_frame_count": source_rgb_preview_unique_frame_count,
        },
        "schema_version": PUBLIC_BUNDLE_SCHEMA_VERSION,
    }
    (destination / "artifacts_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_checksums(destination: Path) -> None:
    checksum_path = destination / "checksums.sha256"
    rows = []
    for path in _sorted_files(destination):
        if path == checksum_path:
            continue
        relative = path.relative_to(destination).as_posix()
        rows.append(f"{_sha256_file(path)}  {relative}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sorted_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def _artifact_kind(relative: str) -> str:
    if relative.startswith("assets/") and relative.lower().endswith(".png"):
        return "visual_preview"
    if relative == "index.html":
        return "html_report"
    if relative == "summary.json":
        return "report_summary"
    if relative.endswith(".csv"):
        return "audit_table"
    if relative.endswith(".md"):
        return "evidence_document"
    if relative.endswith(".json"):
        return "manifest"
    return "artifact"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_residual_local_paths(root: Path) -> None:
    violations = []
    for path in _sorted_files(root):
        if not _is_text_candidate(path):
            continue
        text = path.read_text(encoding="utf-8")
        found = [prefix for prefix in _FORBIDDEN_LOCAL_PREFIXES if prefix in text]
        if found:
            violations.append(
                f"{path.relative_to(root).as_posix()}: {', '.join(found)}"
            )
    if violations:
        joined = "; ".join(violations)
        raise ValueError(f"public report bundle contains residual local paths: {joined}")


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES
