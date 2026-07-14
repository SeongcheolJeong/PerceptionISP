from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from perception_isp.datasets.sid_dataset import build_sid_sony_pairs
from perception_isp.datasets.sid_dataset import _resize_bayer_preserving_pattern


class SIDDatasetTest(unittest.TestCase):
    def test_build_sid_sony_pairs_matches_short_to_long_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "Sony2025.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("Sony/short/00001_00_0.04s.ARW", b"short-dark")
                archive.writestr("Sony/short/00001_00_0.1s.ARW", b"short")
                archive.writestr("Sony/short/00002_00_0.1s.ARW", b"short-2")
                archive.writestr("Sony/long/00001_00_10s.ARW", b"long")
                archive.writestr("Sony/long/00002_00_30s.ARW", b"long-2")

            pairs = build_sid_sony_pairs(archive_path, exposure_s=0.1)

            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0].scene_id, "00001")
            self.assertEqual(pairs[0].short_member, "Sony/short/00001_00_0.1s.ARW")
            self.assertEqual(pairs[0].long_member, "Sony/long/00001_00_10s.ARW")
            self.assertEqual(pairs[1].long_exposure_s, 30.0)

    def test_resize_bayer_preserving_pattern_resizes_each_parity_plane(self) -> None:
        raw = np.zeros((8, 8), dtype=float)
        raw[0::2, 0::2] = 100.0
        raw[0::2, 1::2] = 200.0
        raw[1::2, 0::2] = 300.0
        raw[1::2, 1::2] = 400.0

        resized = _resize_bayer_preserving_pattern(raw, width=4, height=4)

        self.assertEqual(resized.shape, (4, 4))
        self.assertTrue(np.allclose(resized[0::2, 0::2], 100.0))
        self.assertTrue(np.allclose(resized[0::2, 1::2], 200.0))
        self.assertTrue(np.allclose(resized[1::2, 0::2], 300.0))
        self.assertTrue(np.allclose(resized[1::2, 1::2], 400.0))


if __name__ == "__main__":
    unittest.main()
