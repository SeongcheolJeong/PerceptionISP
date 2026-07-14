"""Optional bridge to the CameraE2E/pyisetcam simulator."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from perception_isp.core.numeric import weighted_interpolate
from perception_isp.core.paths import camerae2e_source
from perception_isp.core.synthetic import make_synthetic_raw
from perception_isp.core.types import CalibrationProfile, RawFrame, SensorMetadata


CAMERAE2E_SRC = camerae2e_source()
CAMERAE2E_EVAL_PIXEL_PITCH_M = 2.065e-6
CAMERAE2E_NATIVE_CFA_BRIDGE_VERSION = "native_bayer_v1"
CAMERAE2E_BAYER_CAMERA_TYPES = {
    "RGGB": "bayer-rggb",
    "GRBG": "bayer-grbg",
    "BGGR": "bayer-bggr",
    "GBRG": "bayer-gbrg",
}


def raw_from_camerae2e(
    scene_name: str = "uniform ee",
    width: int = 320,
    height: int = 180,
    cfa_pattern: str = "auto",
) -> RawFrame:
    """Build a RAW-like frame from CameraE2E when its dependencies are usable.

    CameraE2E currently provides an image/sensor simulation stack rather than a
    perception-ISP RAW contract. This bridge extracts the sensor voltage/luma
    where possible and remosaics it into the Perception ISP input shape.
    """

    if str(CAMERAE2E_SRC) not in sys.path:
        sys.path.insert(0, str(CAMERAE2E_SRC))
    try:
        from pyisetcam import AssetStore, camera_compute, camera_create, camera_set, scene_create
    except Exception as exc:  # pragma: no cover - environment-dependent path
        raise RuntimeError(f"CameraE2E import failed: {exc}") from exc

    try:
        store = AssetStore.default()
        scene = scene_create(scene_name, asset_store=store)
        camera_type = _camerae2e_camera_type_for_requested_cfa(cfa_pattern)
        camera = _camera_for_target_resolution(camera_create(camera_type, asset_store=store), camera_set)
        camera = camera_compute(camera, scene, sensor_resize=True, asset_store=store)
        sensor = camera.fields.get("sensor")
        ip = camera.fields.get("ip")
        raw_source = None
        raw_source_key = None
        if sensor is not None:
            if sensor.data.get("volts") is not None:
                raw_source = sensor.data.get("volts")
                raw_source_key = "sensor.volts"
            elif sensor.data.get("dv") is not None:
                raw_source = sensor.data.get("dv")
                raw_source_key = "sensor.dv"
        if raw_source is None and ip is not None:
            if ip.data.get("sensorspace") is not None:
                raw_source = ip.data.get("sensorspace")
                raw_source_key = "ip.sensorspace"
            elif ip.data.get("srgb") is not None:
                raw_source = ip.data.get("srgb")
                raw_source_key = "ip.srgb"
        if raw_source is None:
            raise RuntimeError("CameraE2E produced no sensor or IP image data")
        image = np.asarray(raw_source, dtype=np.float64)
        source_cfa_pattern = _camerae2e_sensor_pattern_name(sensor) if sensor is not None else None
        target_cfa_pattern = _resolve_target_cfa_pattern(cfa_pattern, source_cfa_pattern)
        if image.ndim == 2 and sensor is not None:
            mosaic = _camerae2e_sensor_mosaic_to_pattern(sensor, image, int(height), int(width), target_cfa_pattern)
        elif image.ndim == 2:
            rgb = np.repeat(_resize_image(image, height, width)[:, :, None], 3, axis=2)
            rgb = _normalize(rgb)
            mosaic = _mosaic_from_rgb(rgb, target_cfa_pattern)
        elif image.ndim == 3:
            rgb = _resize_image(image[:, :, :3], height, width)
            rgb = _normalize(rgb)
            mosaic = _mosaic_from_rgb(rgb, target_cfa_pattern)
        else:
            raise RuntimeError(f"Unsupported CameraE2E image shape {image.shape}")
        synthetic = make_synthetic_raw(width=width, height=height, cfa_pattern=target_cfa_pattern)
        raw = _hdr_raw_stack_from_mosaic(mosaic)
        metadata = SensorMetadata(
            camera_id="camerae2e_bridge",
            sensor_id="pyisetcam_sensor",
            module_serial="CameraE2E",
            calibration_id="camerae2e_bridge_calibration",
            isp_profile_id="perception_isp_reference_v1",
            cfa_pattern=target_cfa_pattern,
            hdr_mode="multi_exposure",
            exposure_times_us=(8000.0, 2000.0, 500.0),
            hdr_ratios=(1.0, 0.25, 0.0625),
            line_time_us=33333.0 / float(height),
        )
        return RawFrame(
            data=raw,
            metadata=metadata,
            calibration=synthetic.calibration,
            provenance=_camerae2e_provenance(
                sensor=sensor,
                raw_source=image,
                raw_source_key=raw_source_key,
                target_height=int(height),
                target_width=int(width),
                target_cfa_pattern=target_cfa_pattern,
                requested_cfa_pattern=cfa_pattern,
                scene_source=scene_name,
                camera_type=camera_type,
            ),
        )
    except Exception as exc:  # pragma: no cover - environment-dependent path
        raise RuntimeError(f"CameraE2E simulation failed: {exc}") from exc


def raw_from_camerae2e_rgb(
    rgb: Any,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    cfa_pattern: str = "auto",
    scene_luminance: float = 100.0,
    resize_scene_to_target: bool = True,
) -> RawFrame:
    """Run an RGB array through CameraE2E and convert the result to RAW-like HDR.

    This is the practical path for labeled RGB datasets: keep the original
    annotations, use CameraE2E to simulate the camera pipeline, and remosaic the
    simulated sensor/IP output into PerceptionISP's RAW contract.
    """

    if str(CAMERAE2E_SRC) not in sys.path:
        sys.path.insert(0, str(CAMERAE2E_SRC))
    try:
        from pyisetcam import AssetStore, camera_compute, camera_create, camera_set, scene_from_file
    except Exception as exc:  # pragma: no cover - environment-dependent path
        raise RuntimeError(f"CameraE2E import failed: {exc}") from exc

    try:
        source = np.asarray(rgb, dtype=np.float64)
        if source.ndim != 3 or source.shape[2] < 3:
            raise ValueError("rgb must be an HxWx3 array")
        target_h = int(height) if height is not None else int(source.shape[0])
        target_w = int(width) if width is not None else int(source.shape[1])
        target_rgb = _resize_image(source[:, :, :3], target_h, target_w)
        scene_rgb = target_rgb if bool(resize_scene_to_target) else source[:, :, :3]
        uint8 = np.round(np.clip(_normalize(scene_rgb), 0.0, 1.0) * 255.0).astype(np.uint8)

        store = AssetStore.default()
        scene = scene_from_file(uint8, "rgb", float(scene_luminance), "lcdExample.mat", asset_store=store)
        camera_type = _camerae2e_camera_type_for_requested_cfa(cfa_pattern)
        camera = _camera_for_target_resolution(camera_create(camera_type, asset_store=store), camera_set)
        camera = camera_compute(camera, scene, sensor_resize=True, asset_store=store)
        sensor = camera.fields.get("sensor")
        ip = camera.fields.get("ip")
        raw_source = None
        raw_source_key = None
        if sensor is not None:
            if sensor.data.get("volts") is not None:
                raw_source = sensor.data.get("volts")
                raw_source_key = "sensor.volts"
            elif sensor.data.get("dv") is not None:
                raw_source = sensor.data.get("dv")
                raw_source_key = "sensor.dv"
        if raw_source is None and ip is not None:
            if ip.data.get("sensorspace") is not None:
                raw_source = ip.data.get("sensorspace")
                raw_source_key = "ip.sensorspace"
            elif ip.data.get("srgb") is not None:
                raw_source = ip.data.get("srgb")
                raw_source_key = "ip.srgb"
        if raw_source is None:
            raise RuntimeError("CameraE2E produced no sensor or IP image data")
        image = np.asarray(raw_source, dtype=np.float64)
        source_cfa_pattern = _camerae2e_sensor_pattern_name(sensor) if sensor is not None else None
        target_cfa_pattern = _resolve_target_cfa_pattern(cfa_pattern, source_cfa_pattern)
        resized_rgb = np.clip(_normalize(target_rgb), 0.0, 1.0)
        if image.ndim == 2 and sensor is not None:
            mosaic = _camerae2e_sensor_mosaic_to_pattern(sensor, image, target_h, target_w, target_cfa_pattern)
        elif image.ndim == 3:
            simulated_rgb = _resize_image(image[:, :, :3], target_h, target_w)
            if _is_effectively_monochrome(simulated_rgb):
                simulated_rgb = _apply_simulated_luma_preserve_chroma(resized_rgb, np.mean(simulated_rgb[:, :, :3], axis=2))
            else:
                simulated_rgb = _normalize(simulated_rgb)
            mosaic = _mosaic_from_rgb(simulated_rgb, target_cfa_pattern)
        elif image.ndim == 2:
            simulated_luma = _resize_image(image, target_h, target_w)
            simulated_rgb = _apply_simulated_luma_preserve_chroma(resized_rgb, simulated_luma)
            mosaic = _mosaic_from_rgb(simulated_rgb, target_cfa_pattern)
        else:
            raise RuntimeError(f"Unsupported CameraE2E image shape {image.shape}")
        synthetic = make_synthetic_raw(width=target_w, height=target_h, cfa_pattern=target_cfa_pattern)
        raw = _hdr_raw_stack_from_mosaic(mosaic)
        metadata = SensorMetadata(
            camera_id="camerae2e_rgb_bridge",
            sensor_id="pyisetcam_sensor_from_rgb",
            module_serial="CameraE2E scene_from_file",
            calibration_id="camerae2e_rgb_bridge_calibration",
            isp_profile_id="perception_isp_reference_v1",
            cfa_pattern=target_cfa_pattern,
            hdr_mode="multi_exposure",
            exposure_times_us=(8000.0, 2000.0, 500.0),
            hdr_ratios=(1.0, 0.25, 0.0625),
            line_time_us=33333.0 / float(target_h),
        )
        frame = RawFrame(
            data=raw,
            metadata=metadata,
            calibration=synthetic.calibration,
            provenance=_camerae2e_provenance(
                sensor=sensor,
                raw_source=image,
                raw_source_key=raw_source_key,
                target_height=target_h,
                target_width=target_w,
                target_cfa_pattern=target_cfa_pattern,
                requested_cfa_pattern=cfa_pattern,
                scene_source="rgb_array",
                camera_type=camera_type,
            ),
        )
        frame.provenance["scene_input_shape"] = [int(v) for v in source.shape]
        frame.provenance["scene_resized_to_target"] = bool(resize_scene_to_target)
        return frame
    except Exception as exc:  # pragma: no cover - environment-dependent path
        raise RuntimeError(f"CameraE2E RGB simulation failed: {exc}") from exc


def raw_from_rgb_direct(
    rgb: Any,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    cfa_pattern: str = "RGGB",
) -> RawFrame:
    """Convert an RGB array directly to RAW-like HDR without CameraE2E.

    This is useful for unit tests and for environments where CameraE2E is not
    installed. It is not the preferred path for camera simulation evidence.
    """

    source = np.asarray(rgb, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("rgb must be an HxWx3 array")
    target_h = int(height) if height is not None else int(source.shape[0])
    target_w = int(width) if width is not None else int(source.shape[1])
    target_cfa_pattern = _resolve_target_cfa_pattern(cfa_pattern, None)
    resized = np.clip(_normalize(_resize_image(source[:, :, :3], target_h, target_w)), 0.0, 1.0)
    synthetic = make_synthetic_raw(width=target_w, height=target_h, cfa_pattern=target_cfa_pattern)
    mosaic = _mosaic_from_rgb(resized, target_cfa_pattern)
    raw = np.stack(
        [
            np.round(np.clip(mosaic * 1.0, 0.0, 1.0) * (4095.0 - 64.0) + 64.0),
            np.round(np.clip(mosaic * 0.25, 0.0, 1.0) * (4095.0 - 64.0) + 64.0),
            np.round(np.clip(mosaic * 0.0625, 0.0, 1.0) * (4095.0 - 64.0) + 64.0),
        ],
        axis=0,
    )
    metadata = SensorMetadata(
        camera_id="rgb_direct_bridge",
        sensor_id="direct_rgb_to_raw",
        module_serial="direct RGB remosaic",
        calibration_id="direct_rgb_bridge_calibration",
        isp_profile_id="perception_isp_reference_v1",
        cfa_pattern=target_cfa_pattern,
        hdr_mode="multi_exposure",
        exposure_times_us=(8000.0, 2000.0, 500.0),
        hdr_ratios=(1.0, 0.25, 0.0625),
        line_time_us=33333.0 / float(target_h),
    )
    return RawFrame(
        data=raw,
        metadata=metadata,
        calibration=synthetic.calibration,
        provenance={
            "bridge": "direct_rgb",
            "raw_source_key": "input.rgb",
            "source_shape": [int(source.shape[0]), int(source.shape[1]), int(source.shape[2])],
            "target_shape": [int(target_h), int(target_w)],
            "requested_cfa_pattern": str(cfa_pattern),
            "target_cfa_pattern": target_cfa_pattern,
            "true_sensor_cfa_mosaic": False,
            "native_resolution_matches_target": True,
            "native_resolution_at_least_target": True,
            "camerae2e_used": False,
        },
    )


def camerae2e_or_synthetic_raw(
    use_camerae2e: bool = False,
    scene_name: str = "uniform ee",
    width: int = 320,
    height: int = 180,
    cfa_pattern: str = "auto",
) -> RawFrame:
    """Return CameraE2E RAW if requested and available, otherwise synthetic RAW."""

    if use_camerae2e:
        try:
            return raw_from_camerae2e(scene_name=scene_name, width=width, height=height, cfa_pattern=cfa_pattern)
        except Exception as exc:
            raw = make_synthetic_raw(width=width, height=height, cfa_pattern=_resolve_target_cfa_pattern(cfa_pattern, None))
            raw.metadata = replace(
                raw.metadata,
                camera_id="synthetic_front_fallback",
                sensor_id="synthetic_after_camerae2e_failure",
                module_serial=f"CameraE2E fallback: {exc}",
            )
            return raw
    return make_synthetic_raw(width=width, height=height, cfa_pattern=_resolve_target_cfa_pattern(cfa_pattern, None))


def _camera_for_target_resolution(camera: Any, camera_set_func: Any) -> Any:
    return camera_set_func(camera, "pixel size same fill factor", CAMERAE2E_EVAL_PIXEL_PITCH_M)


def _camerae2e_camera_type_for_requested_cfa(requested_pattern: str) -> str:
    requested = str(requested_pattern or "auto").upper().replace("-", "").replace("_", "")
    if requested in {"AUTO", "SENSOR", "NATIVE", "SOURCE", "CAMERAE2E"}:
        return "default"
    return CAMERAE2E_BAYER_CAMERA_TYPES.get(requested, "default")


def _camerae2e_provenance(
    *,
    sensor: Any,
    raw_source: np.ndarray,
    raw_source_key: Optional[str],
    target_height: int,
    target_width: int,
    target_cfa_pattern: str,
    requested_cfa_pattern: str,
    scene_source: str,
    camera_type: str | None = None,
) -> dict[str, Any]:
    source = np.asarray(raw_source)
    source_pattern = _camerae2e_sensor_pattern_name(sensor) if sensor is not None else None
    source_shape = [int(v) for v in source.shape]
    native_hw = source_shape[:2] if len(source_shape) >= 2 else []
    target_shape = [int(target_height), int(target_width)]
    native_at_least_target = bool(
        len(native_hw) == 2
        and int(native_hw[0]) >= int(target_height)
        and int(native_hw[1]) >= int(target_width)
    )
    return {
        "bridge": "camerae2e",
        "scene_source": str(scene_source),
        "camerae2e_camera_type": str(camera_type or "unknown"),
        "camerae2e_native_cfa_bridge_version": CAMERAE2E_NATIVE_CFA_BRIDGE_VERSION,
        "raw_source_key": str(raw_source_key or "unknown"),
        "source_shape": source_shape,
        "source_native_hw": native_hw,
        "target_shape": target_shape,
        "source_cfa_pattern": source_pattern,
        "requested_cfa_pattern": str(requested_cfa_pattern),
        "target_cfa_pattern": str(target_cfa_pattern),
        "pattern_remapped": bool(source_pattern is not None and str(source_pattern).upper() != str(target_cfa_pattern).upper()),
        "true_sensor_cfa_mosaic": bool(str(raw_source_key or "").startswith("sensor.") and source.ndim == 2),
        "native_resolution_matches_target": bool(native_hw == target_shape),
        "native_resolution_at_least_target": native_at_least_target,
        "camerae2e_used": True,
        "camerae2e_eval_pixel_pitch_m": float(CAMERAE2E_EVAL_PIXEL_PITCH_M),
    }


def _normalize(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if high <= low:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _apply_simulated_luma_preserve_chroma(rgb: np.ndarray, simulated_luma: np.ndarray) -> np.ndarray:
    """Use CameraE2E luma/shading without destroying source chroma.

    CameraE2E sensor voltage is commonly a 2-D RAW/luma plane. For labeled RGB
    datasets, remosaicing that plane directly would create a grayscale RAW and
    make the detector comparison mostly a bridge artifact. This keeps the
    simulator's luminance structure while preserving the scene chroma.
    """

    base = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    luma = _normalize(np.asarray(simulated_luma, dtype=np.float64))
    if luma.ndim == 3:
        luma = np.mean(luma[:, :, :3], axis=2)
    source_luma = _rgb_luma(base)
    scale = luma / np.maximum(source_luma, 1.0e-3)
    scale = np.clip(scale, 0.0, 3.0)
    return np.clip(base * scale[:, :, None], 0.0, 1.0)


def _rgb_luma(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def _is_effectively_monochrome(rgb: np.ndarray) -> bool:
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return True
    spread = np.mean(np.std(arr[:, :, :3], axis=2))
    dynamic = max(float(np.nanmax(arr) - np.nanmin(arr)), 1.0e-6)
    return bool(float(spread) / dynamic < 0.02)


def _camerae2e_sensor_mosaic_to_pattern(sensor: Any, raw_source: Any, height: int, width: int, target_pattern: str) -> np.ndarray:
    source_mosaic = _normalize(np.asarray(raw_source, dtype=np.float64))
    source_pattern = _camerae2e_sensor_pattern_name(sensor)
    target = _resolve_target_cfa_pattern(target_pattern, source_pattern)
    if source_pattern is None:
        return _resize_dense_image(source_mosaic, int(height), int(width))
    if source_mosaic.shape == (int(height), int(width)) and source_pattern == target:
        return np.clip(source_mosaic, 0.0, 1.0)
    dense_rgb = _dense_rgb_from_mosaic(source_mosaic, source_pattern)
    dense_rgb = _resize_dense_image(dense_rgb, int(height), int(width))
    return _mosaic_from_rgb(np.clip(dense_rgb, 0.0, 1.0), target)


def _resolve_target_cfa_pattern(requested_pattern: str, source_pattern: Optional[str]) -> str:
    requested = str(requested_pattern or "auto").upper().replace("-", "").replace("_", "")
    if requested in {"AUTO", "SENSOR", "NATIVE", "SOURCE", "CAMERAE2E"}:
        return str(source_pattern or "RGGB").upper().replace("-", "")
    return str(requested_pattern or "RGGB").upper().replace("-", "")


def _camerae2e_sensor_pattern_name(sensor: Any) -> Optional[str]:
    try:
        from pyisetcam.sensor import sensor_get

        pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
        names = list(sensor_get(sensor, "filter names"))
    except Exception:
        return None
    if pattern.ndim != 2 or pattern.size == 0:
        return None
    letters = []
    for row in range(pattern.shape[0]):
        for col in range(pattern.shape[1]):
            index = int(pattern[row, col]) - 1
            if not 0 <= index < len(names):
                letters.append("G")
                continue
            letters.append(_filter_name_to_cfa_letter(str(names[index])))
    if len(letters) == 4:
        return "".join(letters).upper()
    return None


def _filter_name_to_cfa_letter(name: str) -> str:
    normalized = name.strip().lower()
    if normalized.startswith("r"):
        return "R"
    if normalized.startswith("g"):
        return "G"
    if normalized.startswith("b"):
        return "B"
    if normalized.startswith("c") or normalized.startswith("w"):
        return "C"
    if normalized.startswith("ir") or normalized.startswith("i"):
        return "IR"
    return "G"


def _dense_rgb_from_mosaic(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    masks = _local_cfa_masks(pattern, mosaic.shape)
    r = weighted_interpolate(mosaic * masks["R"], masks["R"], radius=2)
    g = weighted_interpolate(mosaic * masks["G"], masks["G"], radius=1)
    b = weighted_interpolate(mosaic * masks["B"], masks["B"], radius=2)
    return np.stack([r, g, b], axis=2)


def _local_cfa_masks(pattern: str, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    rows, cols = int(shape[0]), int(shape[1])
    pattern = str(pattern or "RGGB").upper().replace("-", "")
    if pattern == "RGGB":
        tile = (("R", "G"), ("G", "B"))
    elif pattern == "BGGR":
        tile = (("B", "G"), ("G", "R"))
    elif pattern == "GRBG":
        tile = (("G", "R"), ("B", "G"))
    elif pattern == "GBRG":
        tile = (("G", "B"), ("R", "G"))
    else:
        tile = (("R", "G"), ("G", "B"))
    masks = {name: np.zeros((rows, cols), dtype=np.float64) for name in ("R", "G", "B")}
    for row in range(2):
        for col in range(2):
            masks[tile[row][col]][row::2, col::2] = 1.0
    return masks


def _resize_dense_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.shape[:2] == (int(height), int(width)):
        return arr.copy()
    try:
        from PIL import Image

        if arr.ndim == 2:
            pil = Image.fromarray(arr.astype(np.float32))
            return np.asarray(pil.resize((int(width), int(height)), resample=Image.Resampling.BILINEAR), dtype=np.float64)
        channels = [
            np.asarray(
                Image.fromarray(arr[:, :, index].astype(np.float32)).resize((int(width), int(height)), resample=Image.Resampling.BILINEAR),
                dtype=np.float64,
            )
            for index in range(arr.shape[2])
        ]
        return np.stack(channels, axis=2)
    except Exception:
        return _resize_image(arr, int(height), int(width))


def _hdr_raw_stack_from_mosaic(mosaic: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(mosaic, dtype=np.float64), 0.0, 1.0)
    return np.stack(
        [
            np.round(np.clip(values * 1.0, 0.0, 1.0) * (4095.0 - 64.0) + 64.0),
            np.round(np.clip(values * 0.25, 0.0, 1.0) * (4095.0 - 64.0) + 64.0),
            np.round(np.clip(values * 0.0625, 0.0, 1.0) * (4095.0 - 64.0) + 64.0),
        ],
        axis=0,
    )


def _resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    y = np.clip(np.round(np.linspace(0, arr.shape[0] - 1, int(height))).astype(int), 0, arr.shape[0] - 1)
    x = np.clip(np.round(np.linspace(0, arr.shape[1] - 1, int(width))).astype(int), 0, arr.shape[1] - 1)
    if arr.ndim == 2:
        return arr[np.ix_(y, x)]
    return arr[np.ix_(y, x, np.arange(arr.shape[2]))]


def _mosaic_from_rgb(rgb: np.ndarray, pattern: str) -> np.ndarray:
    from perception_isp.core.synthetic import _mosaic

    return _mosaic(np.asarray(rgb, dtype=np.float64), pattern)
