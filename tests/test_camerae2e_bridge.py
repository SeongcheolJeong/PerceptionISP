from __future__ import annotations

import unittest

import numpy as np

from perception_isp.camerae2e_bridge import _resolve_target_cfa_pattern, raw_from_rgb_direct


class CameraE2EBridgeTest(unittest.TestCase):
    def test_auto_cfa_uses_source_pattern_when_available(self) -> None:
        self.assertEqual(_resolve_target_cfa_pattern("auto", "GRBG"), "GRBG")
        self.assertEqual(_resolve_target_cfa_pattern("native", "GBRG"), "GBRG")
        self.assertEqual(_resolve_target_cfa_pattern("RGGB", "GRBG"), "RGGB")

    def test_auto_cfa_falls_back_to_rggb_without_sensor_source(self) -> None:
        rgb = np.ones((24, 32, 3), dtype=np.float64) * 0.5
        raw = raw_from_rgb_direct(rgb, cfa_pattern="auto")
        self.assertEqual(raw.metadata.cfa_pattern, "RGGB")
        self.assertEqual(raw.provenance["requested_cfa_pattern"], "auto")
        self.assertEqual(raw.provenance["target_cfa_pattern"], "RGGB")


if __name__ == "__main__":
    unittest.main()
