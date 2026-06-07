from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from perception_isp.camerae2e_bridge import (
    CAMERAE2E_SRC,
    _camerae2e_provenance,
    _camerae2e_camera_type_for_requested_cfa,
    _camerae2e_sensor_mosaic_to_pattern,
    _camerae2e_sensor_pattern_name,
    _resolve_target_cfa_pattern,
    raw_from_camerae2e_rgb,
    raw_from_rgb_direct,
)


class CameraE2EBridgeTest(unittest.TestCase):
    def test_auto_cfa_uses_source_pattern_when_available(self) -> None:
        self.assertEqual(_resolve_target_cfa_pattern("auto", "GRBG"), "GRBG")
        self.assertEqual(_resolve_target_cfa_pattern("native", "GBRG"), "GBRG")
        self.assertEqual(_resolve_target_cfa_pattern("RGGB", "GRBG"), "RGGB")

    def test_explicit_bayer_cfa_selects_camerae2e_camera_type(self) -> None:
        self.assertEqual(_camerae2e_camera_type_for_requested_cfa("auto"), "default")
        self.assertEqual(_camerae2e_camera_type_for_requested_cfa("native"), "default")
        self.assertEqual(_camerae2e_camera_type_for_requested_cfa("RGGB"), "bayer-rggb")
        self.assertEqual(_camerae2e_camera_type_for_requested_cfa("GRBG"), "bayer-grbg")
        self.assertEqual(_camerae2e_camera_type_for_requested_cfa("BGGR"), "bayer-bggr")
        self.assertEqual(_camerae2e_camera_type_for_requested_cfa("GBRG"), "bayer-gbrg")
        self.assertEqual(_camerae2e_camera_type_for_requested_cfa("RCCB"), "default")

    def test_auto_cfa_falls_back_to_rggb_without_sensor_source(self) -> None:
        rgb = np.ones((24, 32, 3), dtype=np.float64) * 0.5
        raw = raw_from_rgb_direct(rgb, cfa_pattern="auto")
        self.assertEqual(raw.metadata.cfa_pattern, "RGGB")
        self.assertEqual(raw.provenance["requested_cfa_pattern"], "auto")
        self.assertEqual(raw.provenance["target_cfa_pattern"], "RGGB")

    def test_camerae2e_pattern_name_maps_sensor_pattern_to_bayer_order(self) -> None:
        sensor = _FakeSensor(pattern=np.array([[2, 1], [3, 2]], dtype=int), filter_names=["r", "g", "b"])

        with patch.dict(sys.modules, _fake_pyisetcam_modules()):
            self.assertEqual(_camerae2e_sensor_pattern_name(sensor), "GRBG")

    def test_native_sensor_mosaic_is_preserved_when_pattern_and_shape_match(self) -> None:
        sensor = _FakeSensor(pattern=np.array([[2, 1], [3, 2]], dtype=int), filter_names=["r", "g", "b"])
        mosaic = np.linspace(0.0, 1.0, 48, dtype=np.float64).reshape(6, 8)

        with patch.dict(sys.modules, _fake_pyisetcam_modules()):
            converted = _camerae2e_sensor_mosaic_to_pattern(sensor, mosaic, height=6, width=8, target_pattern="auto")

        np.testing.assert_allclose(converted, mosaic)

    def test_camerae2e_provenance_flags_true_native_cfa_and_remap(self) -> None:
        sensor = _FakeSensor(pattern=np.array([[2, 1], [3, 2]], dtype=int), filter_names=["r", "g", "b"])
        source = np.zeros((48, 64), dtype=np.float64)

        with patch.dict(sys.modules, _fake_pyisetcam_modules()):
            native = _camerae2e_provenance(
                sensor=sensor,
                raw_source=source,
                raw_source_key="sensor.volts",
                target_height=48,
                target_width=64,
                target_cfa_pattern="GRBG",
                requested_cfa_pattern="auto",
                scene_source="unit",
            )
            remapped = _camerae2e_provenance(
                sensor=sensor,
                raw_source=source,
                raw_source_key="sensor.volts",
                target_height=48,
                target_width=64,
                target_cfa_pattern="RGGB",
                requested_cfa_pattern="RGGB",
                scene_source="unit",
            )

        self.assertTrue(native["true_sensor_cfa_mosaic"])
        self.assertTrue(native["native_resolution_matches_target"])
        self.assertTrue(native["native_resolution_at_least_target"])
        self.assertFalse(native["pattern_remapped"])
        self.assertEqual(native["source_cfa_pattern"], "GRBG")
        self.assertEqual(native["target_cfa_pattern"], "GRBG")
        self.assertTrue(remapped["pattern_remapped"])
        self.assertEqual(remapped["target_cfa_pattern"], "RGGB")
        self.assertEqual(native["camerae2e_native_cfa_bridge_version"], "native_bayer_v1")

    def test_camerae2e_rgb_runtime_uses_sensor_native_cfa_when_auto(self) -> None:
        if not _camerae2e_runtime_available():
            self.skipTest("CameraE2E runtime dependencies are not available")
        rgb = np.zeros((24, 32, 3), dtype=np.float64)
        rgb[:, :16, 0] = 1.0
        rgb[:, 16:, 1] = 1.0

        raw = raw_from_camerae2e_rgb(rgb, width=32, height=24, cfa_pattern="auto")

        self.assertEqual(raw.data.shape, (3, 24, 32))
        self.assertEqual(raw.metadata.cfa_pattern, raw.provenance["source_cfa_pattern"])
        self.assertEqual(raw.provenance["raw_source_key"], "sensor.volts")
        self.assertTrue(raw.provenance["true_sensor_cfa_mosaic"])
        self.assertFalse(raw.provenance["pattern_remapped"])
        self.assertTrue(raw.provenance["native_resolution_matches_target"])

    def test_camerae2e_rgb_runtime_uses_requested_native_bayer_cfa(self) -> None:
        if not _camerae2e_runtime_available():
            self.skipTest("CameraE2E runtime dependencies are not available")
        rgb = np.zeros((24, 32, 3), dtype=np.float64)
        rgb[:, :16, 0] = 1.0
        rgb[:, 16:, 2] = 1.0

        raw = raw_from_camerae2e_rgb(rgb, width=32, height=24, cfa_pattern="RGGB")

        self.assertEqual(raw.data.shape, (3, 24, 32))
        self.assertEqual(raw.metadata.cfa_pattern, "RGGB")
        self.assertEqual(raw.provenance["source_cfa_pattern"], "RGGB")
        self.assertEqual(raw.provenance["target_cfa_pattern"], "RGGB")
        self.assertEqual(raw.provenance["camerae2e_camera_type"], "bayer-rggb")
        self.assertTrue(raw.provenance["true_sensor_cfa_mosaic"])
        self.assertFalse(raw.provenance["pattern_remapped"])
        self.assertTrue(raw.provenance["native_resolution_matches_target"])


class _FakeSensor:
    def __init__(self, *, pattern: np.ndarray, filter_names: list[str]) -> None:
        self.pattern = np.asarray(pattern, dtype=int)
        self.filter_names = list(filter_names)


def _fake_pyisetcam_modules() -> dict[str, types.ModuleType]:
    package = types.ModuleType("pyisetcam")
    package.__path__ = []  # type: ignore[attr-defined]
    sensor_module = types.ModuleType("pyisetcam.sensor")

    def sensor_get(sensor: _FakeSensor, parameter: str):
        normalized = str(parameter).strip().lower()
        if normalized == "pattern":
            return sensor.pattern
        if normalized == "filter names":
            return sensor.filter_names
        raise KeyError(parameter)

    sensor_module.sensor_get = sensor_get  # type: ignore[attr-defined]
    return {"pyisetcam": package, "pyisetcam.sensor": sensor_module}


def _camerae2e_runtime_available() -> bool:
    if not CAMERAE2E_SRC.exists():
        return False
    if str(CAMERAE2E_SRC) not in sys.path:
        sys.path.insert(0, str(CAMERAE2E_SRC))
    try:
        import h5py  # noqa: F401
        import pyisetcam  # noqa: F401
    except Exception:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
