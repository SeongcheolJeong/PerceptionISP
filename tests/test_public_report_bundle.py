from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from perception_isp.reporting.public_bundle import (
    PUBLIC_BUNDLE_SCHEMA_VERSION,
    SHOWCASE_SOURCE_FILES,
    build_public_report_bundle,
)


class PublicReportBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.status = self.root / "status-report"
        self.showcase = self.root / "showcase-report"
        self.accomplishment = self.root / "accomplishment-report"
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_deterministic_lightweight_bundle_without_mutating_sources(self) -> None:
        original_showcase_html = (self.showcase / "index.html").read_bytes()
        original_stale_manifest = (self.showcase / "artifacts_manifest.json").read_bytes()
        original_source_rgb = (
            self.showcase / "assets" / "temporal_frame_000_source_jpeg_rgb.png"
        ).read_bytes()

        first = build_public_report_bundle(
            status_report_dir=self.status,
            showcase_report_dir=self.showcase,
            destination=self.root / "public-a",
            local_prefix_redactions=self._redactions(),
            extra_report_dirs=(self.accomplishment,),
        )
        second = build_public_report_bundle(
            status_report_dir=self.status,
            showcase_report_dir=self.showcase,
            destination=self.root / "public-b",
            local_prefix_redactions=self._redactions(),
            extra_report_dirs=(self.accomplishment,),
        )
        republished = build_public_report_bundle(
            status_report_dir=first.root / "status-report",
            showcase_report_dir=first.root / "showcase-report",
            destination=self.root / "public-from-public",
            extra_report_dirs=(first.root / "accomplishment-report",),
        )

        self.assertEqual(first.status_report, first.root / "status-report" / "index.html")
        self.assertEqual(
            first.showcase_report, first.root / "showcase-report" / "index.html"
        )
        self.assertEqual(
            first.extra_reports,
            (first.root / "accomplishment-report" / "index.html",),
        )
        self.assertEqual(self._file_payloads(first.root), self._file_payloads(second.root))
        self.assertEqual(
            self._file_payloads(first.root), self._file_payloads(republished.root)
        )

        published_showcase = first.root / "showcase-report"
        self.assertFalse((published_showcase / "arrays").exists())
        self.assertFalse((published_showcase / "assets" / "debug.npy").exists())
        self.assertEqual(
            (published_showcase / "assets" / "preview.png").read_bytes(), b"PNG-one"
        )
        self.assertEqual(
            (published_showcase / "assets" / "nested" / "map.PNG").read_bytes(),
            b"PNG-two",
        )
        with Image.open(
            published_showcase / "assets" / "temporal_frame_000_source_jpeg_rgb.png"
        ) as source_rgb_preview:
            self.assertEqual(source_rgb_preview.size, (800, 400))
        for name in SHOWCASE_SOURCE_FILES:
            self.assertTrue((published_showcase / name).is_file(), name)

        html = (published_showcase / "index.html").read_text(encoding="utf-8")
        evidence = (published_showcase / "EVIDENCE.md").read_text(encoding="utf-8")
        notice = (published_showcase / "DATA_NOTICE.md").read_text(encoding="utf-8")
        self.assertNotIn("arrays/", html)
        self.assertNotIn("lossless trace", html.lower())
        self.assertIn("public-preview textual provenance", html)
        self.assertIn("lossless numerical arrays are intentionally omitted", html)
        self.assertIn(
            "Source-RGB input thumbnails are bounded to 800px (2 published paths; "
            "unique frame images: 1)",
            html,
        )
        self.assertIn("result PNGs are copied byte-for-byte", html)
        self.assertIn("href='DATA_NOTICE.md'", html)
        self.assertEqual(html.count("public-bundle-notice"), 1)
        self.assertNotIn("`arrays/`", evidence)
        self.assertNotIn("Numerical verification must use the NPY", evidence)
        self.assertIn("[DATA_NOTICE.md](DATA_NOTICE.md)", evidence)
        self.assertIn(
            "The 2 published source-RGB paths have a unique frame-image count of 1",
            evidence,
        )
        self.assertNotRegex(evidence, r"(?m)[ \t]+$")
        self.assertIn("CC BY-NC-SA 4.0", notice)
        self.assertIn("nuScenes terms of use", notice)
        self.assertIn("arXiv:1903.11027", notice)
        self.assertIn("do not endorse", notice)
        self.assertIn("Published PNG previews: **4**", notice)
        self.assertIn("Downscaled source-RGB input previews: **2**", notice)
        self.assertIn("Source-RGB preview paths: **2**", notice)
        self.assertIn("Unique source-frame images represented by those paths: **1**", notice)
        self.assertIn("Aux-map, pseudo-RAW, and PerceptionISP result PNGs", notice)

        public_summary = json.loads(
            (published_showcase / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(public_summary["publication_profile"], "github_preview")
        self.assertTrue(public_summary["preview_only"])
        self.assertFalse(public_summary["arrays_included"])
        self.assertEqual(public_summary["data_notice"], "DATA_NOTICE.md")
        self.assertEqual(public_summary["source_rgb_preview_downscaled_count"], 2)
        self.assertEqual(public_summary["source_rgb_preview_max_edge_px"], 800)
        self.assertEqual(public_summary["source_rgb_preview_path_count"], 2)
        self.assertEqual(public_summary["source_rgb_preview_unique_frame_count"], 1)
        self.assertIn("intentionally omitted", public_summary["interpretation"])
        self.assertNotIn(
            "Every displayed and lossless derived artifact",
            public_summary["interpretation"],
        )

        status_text = (first.root / "status-report" / "index.html").read_text(
            encoding="utf-8"
        )
        status_json = (first.root / "status-report" / "project_status_summary.json").read_text(
            encoding="utf-8"
        )
        accomplishment_json = (
            first.root / "accomplishment-report" / "project_accomplishment_summary.json"
        ).read_text(encoding="utf-8")
        self.assertIn("${PROJECT_ROOT}", status_text)
        self.assertIn("../../docs/ARCHITECTURE.md", status_text)
        self.assertIn("${CAMERAE2E_ROOT}", status_json)
        self.assertIn("${PROJECT_ROOT}", accomplishment_json)
        self.assertTrue((first.root / "accomplishment-report" / "chart.png").is_file())
        self.assertFalse((first.root / "accomplishment-report" / "arrays").exists())
        self.assertFalse(
            (first.root / "accomplishment-report" / "checksums.sha256").exists()
        )

        for path in self._text_files(first.root):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", text, path)
            self.assertNotIn("/tmp/", text, path)

        self.assertEqual((self.showcase / "index.html").read_bytes(), original_showcase_html)
        self.assertEqual(
            (self.showcase / "artifacts_manifest.json").read_bytes(),
            original_stale_manifest,
        )
        self.assertEqual(
            (self.showcase / "assets" / "temporal_frame_000_source_jpeg_rgb.png").read_bytes(),
            original_source_rgb,
        )

    def test_regenerates_filtered_manifest_and_checksums(self) -> None:
        result = build_public_report_bundle(
            status_report_dir=self.status,
            showcase_report_dir=self.showcase,
            destination=self.root / "public",
            local_prefix_redactions=self._redactions(),
        )
        showcase = result.showcase_report.parent
        manifest_path = showcase / "artifacts_manifest.json"
        checksum_path = showcase / "checksums.sha256"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], PUBLIC_BUNDLE_SCHEMA_VERSION)
        self.assertFalse(manifest["notes"]["arrays_included"])
        self.assertEqual(manifest["notes"]["source_rgb_preview_downscaled_count"], 2)
        self.assertEqual(manifest["notes"]["source_rgb_preview_max_edge_px"], 800)
        self.assertEqual(manifest["notes"]["source_rgb_preview_path_count"], 2)
        self.assertEqual(manifest["notes"]["source_rgb_preview_unique_frame_count"], 1)
        artifact_paths = [row["path"] for row in manifest["artifacts"]]
        self.assertEqual(artifact_paths, sorted(artifact_paths))
        self.assertNotIn("artifacts_manifest.json", artifact_paths)
        self.assertNotIn("checksums.sha256", artifact_paths)
        self.assertFalse(any(path.startswith("arrays/") for path in artifact_paths))
        self.assertEqual(manifest["artifact_count"], len(artifact_paths))
        self.assertIn("DATA_NOTICE.md", artifact_paths)
        self.assertIn("assets/preview.png", artifact_paths)
        for row in manifest["artifacts"]:
            artifact = showcase / row["path"]
            self.assertEqual(row["size_bytes"], artifact.stat().st_size)
            self.assertEqual(row["sha256"], self._sha256(artifact))

        checksum_rows = checksum_path.read_text(encoding="utf-8").splitlines()
        checksum_paths = [row.split("  ", 1)[1] for row in checksum_rows]
        self.assertEqual(checksum_paths, sorted(checksum_paths))
        self.assertIn("artifacts_manifest.json", checksum_paths)
        self.assertNotIn("checksums.sha256", checksum_paths)
        self.assertFalse(any(path.startswith("arrays/") for path in checksum_paths))
        for row in checksum_rows:
            expected, relative = row.split("  ", 1)
            self.assertEqual(expected, self._sha256(showcase / relative))

    def test_rejects_unscrubbed_local_paths_without_leaving_destination(self) -> None:
        destination = self.root / "rejected-public"
        with self.assertRaisesRegex(ValueError, "residual local paths"):
            build_public_report_bundle(
                status_report_dir=self.status,
                showcase_report_dir=self.showcase,
                destination=destination,
                local_prefix_redactions={"/Users/alice/project": "${PROJECT_ROOT}"},
            )

        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".rejected-public.building-*")), [])

    def test_rejects_existing_destination(self) -> None:
        destination = self.root / "already-there"
        destination.mkdir()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            build_public_report_bundle(
                status_report_dir=self.status,
                showcase_report_dir=self.showcase,
                destination=destination,
                local_prefix_redactions=self._redactions(),
            )

    def _write_fixture(self) -> None:
        self.status.mkdir()
        (self.status / "index.html").write_text(
            "<html><body>/Users/alice/project "
            "<a href='../showcase-report/index.html'>showcase</a> "
            "<a href='../accomplishment-report/index.html'>history</a> "
            "<a href='../../docs/ARCHITECTURE.md'>architecture</a></body></html>",
            encoding="utf-8",
        )
        (self.status / "project_status_summary.json").write_text(
            json.dumps(
                {
                    "project": "/Users/alice/project",
                    "camera": "/Users/alice/camera",
                }
            ),
            encoding="utf-8",
        )

        self.showcase.mkdir()
        (self.showcase / "index.html").write_text(
            "<html><body><p>Every result has RGB, 33 maps, and lossless trace.</p>"
            "<p>Lossless 배열은 <code>arrays/</code>, 실행 환경과 hash는 "
            "evidence manifest에 저장됩니다.</p><div class='evidence-links'>"
            "<a href='EVIDENCE.md'>EVIDENCE.md</a></div>"
            "/Users/alice/project/data/nuscenes</body></html>",
            encoding="utf-8",
        )
        (self.showcase / "EVIDENCE.md").write_text(
            "# Evidence\n\n"
            "Command: /Users/alice/camera/python --output /tmp/run-123\n\n"
            "- `arrays/`: lossless NPY copies.\n"
            "- PNG files are visual previews. Numerical verification must use the NPY arrays.\n",
            encoding="utf-8",
        )
        text_payloads = {
            "aux_map_catalog.json": "{}\n",
            "environment.json": '{"cwd": "/Users/alice/project"}\n',
            "exposure_plane_sources.csv": "path\n/tmp/run-123/frame\n",
            "metadata_origins.csv": "field,origin\nexposure,unknown\n",
            "source_manifest.json": '{"path": "/Users/alice/project/data/frame.jpg"}\n',
            "summary.json": (
                '{"camera": "/Users/alice/camera", "output": "/tmp/run-123", '
                '"interpretation": "Every displayed and lossless derived artifact exists."}\n'
            ),
        }
        for name, payload in text_payloads.items():
            (self.showcase / name).write_text(payload, encoding="utf-8")
        (self.showcase / "artifacts_manifest.json").write_text(
            '{"stale": "arrays/all.npy"}\n', encoding="utf-8"
        )
        (self.showcase / "checksums.sha256").write_text(
            "stale  arrays/all.npy\n", encoding="utf-8"
        )
        (self.showcase / "arrays").mkdir()
        (self.showcase / "arrays" / "all.npy").write_bytes(b"large-array")
        (self.showcase / "assets" / "nested").mkdir(parents=True)
        (self.showcase / "assets" / "preview.png").write_bytes(b"PNG-one")
        (self.showcase / "assets" / "nested" / "map.PNG").write_bytes(b"PNG-two")
        Image.new("RGB", (1200, 600), color=(32, 64, 96)).save(
            self.showcase / "assets" / "temporal_frame_000_source_jpeg_rgb.png"
        )
        (self.showcase / "assets" / "interframe_cycle_000_frame_000_source_jpeg_rgb.png").write_bytes(
            (self.showcase / "assets" / "temporal_frame_000_source_jpeg_rgb.png").read_bytes()
        )
        (self.showcase / "assets" / "debug.npy").write_bytes(b"not-public")

        self.accomplishment.mkdir()
        (self.accomplishment / "index.html").write_text(
            "<html><body>Historical report</body></html>", encoding="utf-8"
        )
        (self.accomplishment / "project_accomplishment_summary.json").write_text(
            '{"project_root": "/Users/alice/project"}\n', encoding="utf-8"
        )
        (self.accomplishment / "chart.png").write_bytes(b"PNG-chart")
        (self.accomplishment / "checksums.sha256").write_text(
            "stale\n", encoding="utf-8"
        )
        (self.accomplishment / "arrays").mkdir()
        (self.accomplishment / "arrays" / "ignored.npy").write_bytes(b"ignored")

    @staticmethod
    def _redactions() -> dict[str, str]:
        return {
            "/Users/alice/project": "${PROJECT_ROOT}",
            "/Users/alice/camera": "${CAMERAE2E_ROOT}",
            "/tmp/run-123": "${REPORT_OUTPUT}",
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _file_payloads(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    @staticmethod
    def _text_files(root: Path) -> list[Path]:
        suffixes = {".css", ".csv", ".html", ".js", ".json", ".md", ".sha256"}
        return [path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes]


if __name__ == "__main__":
    unittest.main()
