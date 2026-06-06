"""Core data contracts for the software Perception ISP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class SensorMetadata:
    """Frame metadata that must travel with the DNN-facing tensors."""

    camera_id: str = "camera_0"
    sensor_id: str = "sensor_unknown"
    module_serial: str = "module_unknown"
    calibration_id: str = "calibration_default"
    isp_profile_id: str = "perception_isp_default"
    frame_counter: int = 0
    timestamp_us: float = 0.0
    exposure_times_us: Tuple[float, ...] = (8000.0,)
    analog_gains: Tuple[float, ...] = (1.0,)
    digital_gains: Tuple[float, ...] = (1.0,)
    temperature_c: float = 25.0
    hdr_mode: str = "single"
    hdr_ratios: Tuple[float, ...] = (1.0,)
    cfa_pattern: str = "RGGB"
    readout_direction: str = "top_to_bottom"
    rolling_shutter_time_us: float = 33333.0
    line_time_us: float = 30.864
    lens_profile_id: str = "lens_default"
    color_profile_id: str = "color_default"
    noise_model_id: str = "noise_default"


@dataclass
class CalibrationProfile:
    """Model-level, unit-level, and runtime calibration payload.

    Array fields may be supplied at full resolution or as small maps; the
    pipeline resizes them with nearest-neighbor sampling.
    """

    cfa_pattern: str = "RGGB"
    black_level: float = 64.0
    white_level: float = 4095.0
    companding_gamma: float = 1.0
    defect_pixels: Tuple[Tuple[int, int], ...] = ()
    fpn_offset: Optional[ArrayF] = None
    prnu_gain: Optional[ArrayF] = None
    dsnu_offset: Optional[ArrayF] = None
    lens_shading_gain: Optional[ArrayF] = None
    color_shading_gain: Optional[ArrayF] = None
    color_matrix: ArrayF = field(default_factory=lambda: np.eye(3, dtype=float))
    perception_color_matrix: ArrayF = field(default_factory=lambda: np.eye(3, dtype=float))
    rgb_ir_crosstalk: ArrayF = field(
        default_factory=lambda: np.array(
            [
                [1.0, 0.02, 0.02, -0.06],
                [0.02, 1.0, 0.02, -0.04],
                [0.02, 0.02, 1.0, -0.07],
            ],
            dtype=float,
        )
    )
    shot_noise_coeff: float = 0.0025
    read_noise_var: float = 0.00008
    dark_current_coeff: float = 0.000001
    quantization_var: float = 1.0 / (12.0 * 4095.0 * 4095.0)
    calibration_residual_var: float = 0.00002
    mtf_confidence_map: Optional[ArrayF] = None
    psf_sigma_map: Optional[ArrayF] = None
    intrinsic_matrix: ArrayF = field(default_factory=lambda: np.eye(3, dtype=float))
    distortion_coeffs: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    extrinsic_matrix: ArrayF = field(default_factory=lambda: np.eye(4, dtype=float))


@dataclass(frozen=True)
class PerceptionISPConfig:
    """Runtime tuning for the reference pipeline."""

    output_bit_depth: int = 12
    hdr_saturation_threshold: float = 0.985
    hdr_low_signal_threshold: float = 0.015
    denoise_strength: float = 0.18
    edge_confidence_floor: float = 0.05
    fast_path_roi: str = "bottom"
    fast_path_fraction: float = 1.0 / 3.0
    fast_path_stripe_height: int = 128
    fast_path_low_res_factor: int = 4
    max_edge_packets: int = 512
    edge_packet_threshold: float = 0.12
    accurate_enable_dewarp: bool = False
    tone_mapping: str = "log"
    demosaic_method: str = "edge_aware"
    demosaic_artifact_suppression: float = 0.20
    gamma: float = 2.2
    include_human_view: bool = True
    include_raw_like_tensor: bool = True
    temporal_flicker_threshold: float = 0.15
    blur_edge_percentile: float = 90.0


@dataclass
class RawFrame:
    """Input RAW frame.

    ``data`` accepts either ``H x W`` single-exposure RAW, ``E x H x W`` HDR RAW,
    or ``H x W x E`` HDR RAW.
    """

    data: NDArray[Any]
    metadata: SensorMetadata = field(default_factory=SensorMetadata)
    calibration: CalibrationProfile = field(default_factory=CalibrationProfile)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PreviousFrameState:
    """Small temporal state passed from the previous processed frame."""

    luma: Optional[ArrayF] = None
    rgb: Optional[ArrayF] = None
    timestamp_us: float = 0.0
    frame_counter: int = -1


@dataclass
class EdgePacket:
    """Sparse fast-path edge primitive."""

    row: int
    col: int
    edge_strength: float
    edge_orientation_rad: float
    edge_confidence: float
    noise_confidence: float
    hdr_source: int
    saturation_state: float
    motion_consistency: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row": int(self.row),
            "col": int(self.col),
            "edge_strength": float(self.edge_strength),
            "edge_orientation_rad": float(self.edge_orientation_rad),
            "edge_confidence": float(self.edge_confidence),
            "noise_confidence": float(self.noise_confidence),
            "hdr_source": int(self.hdr_source),
            "saturation_state": float(self.saturation_state),
            "motion_consistency": float(self.motion_consistency),
        }


@dataclass
class FastPathOutput:
    """Low-latency stripe/ROI output."""

    tensor: ArrayF
    channels: Tuple[str, ...]
    roi: Tuple[int, int, int, int]
    edge_packets: Tuple[EdgePacket, ...]
    estimated_latency_us: float
    metadata: Mapping[str, Any]


@dataclass
class AccuratePathOutput:
    """Full-frame perception output."""

    tensor: ArrayF
    channels: Tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass
class PerceptionISPResult:
    """Complete output of the software Perception ISP."""

    human_rgb: Optional[ArrayF]
    vision_rgb: ArrayF
    raw_normalized: ArrayF
    accurate: AccuratePathOutput
    fast: FastPathOutput
    maps: Mapping[str, ArrayF]
    metadata: Mapping[str, Any]
    health: Mapping[str, Any]
    next_state: PreviousFrameState


@dataclass(frozen=True)
class RuntimeControlSuggestion:
    """Scene-adaptive ISP control suggestion."""

    exposure_priority: str
    hdr_priority: float
    denoise_strength: float
    fast_path_priority: float
    enable_dewarp_fast_path: bool
    notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exposure_priority": self.exposure_priority,
            "hdr_priority": float(self.hdr_priority),
            "denoise_strength": float(self.denoise_strength),
            "fast_path_priority": float(self.fast_path_priority),
            "enable_dewarp_fast_path": bool(self.enable_dewarp_fast_path),
            "notes": list(self.notes),
        }


def metadata_to_dict(metadata: SensorMetadata) -> Dict[str, Any]:
    return {
        "camera_id": metadata.camera_id,
        "sensor_id": metadata.sensor_id,
        "module_serial": metadata.module_serial,
        "calibration_id": metadata.calibration_id,
        "isp_profile_id": metadata.isp_profile_id,
        "frame_counter": int(metadata.frame_counter),
        "timestamp_us": float(metadata.timestamp_us),
        "exposure_times_us": [float(v) for v in metadata.exposure_times_us],
        "analog_gains": [float(v) for v in metadata.analog_gains],
        "digital_gains": [float(v) for v in metadata.digital_gains],
        "temperature_c": float(metadata.temperature_c),
        "hdr_mode": metadata.hdr_mode,
        "hdr_ratios": [float(v) for v in metadata.hdr_ratios],
        "cfa_pattern": metadata.cfa_pattern,
        "readout_direction": metadata.readout_direction,
        "rolling_shutter_time_us": float(metadata.rolling_shutter_time_us),
        "line_time_us": float(metadata.line_time_us),
        "lens_profile_id": metadata.lens_profile_id,
        "color_profile_id": metadata.color_profile_id,
        "noise_model_id": metadata.noise_model_id,
    }


def json_ready(value: Any) -> Any:
    """Convert dataclasses, numpy arrays, and scalars into JSON-safe payloads."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, EdgePacket):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value
