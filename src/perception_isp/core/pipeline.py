"""Software reference Perception ISP pipeline.

The implementation is intentionally explicit: each method maps to a block in
the architecture document and returns concrete image/map outputs. It is not a
product ISP, but it is a runnable algorithmic baseline for SW experiments.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from perception_isp.core.numeric import (
    EPS,
    apply_matrix_rgb,
    as_float_array,
    block_reduce_mean,
    box_filter,
    clip01,
    edge_aware_denoise,
    ensure_exposure_first,
    gamma_encode,
    gradient_xy,
    log_tonemap,
    nearest_dewarp,
    percentile,
    resize_nearest,
    row_timestamp_map,
    safe_log_ratio,
    weighted_interpolate,
)
from perception_isp.core.types import (
    AccuratePathOutput,
    ArrayF,
    CalibrationProfile,
    EdgePacket,
    FastPathOutput,
    PerceptionISPConfig,
    PerceptionISPResult,
    PreviousFrameState,
    RawFrame,
    RuntimeControlSuggestion,
    SensorMetadata,
    metadata_to_dict,
)


class PerceptionISPPipeline:
    """Full software Perception ISP reference pipeline."""

    def __init__(
        self,
        config: Optional[PerceptionISPConfig] = None,
        calibration: Optional[CalibrationProfile] = None,
    ) -> None:
        self.config = config or PerceptionISPConfig()
        self.default_calibration = calibration or CalibrationProfile()

    def run(
        self,
        raw: RawFrame,
        previous_state: Optional[PreviousFrameState] = None,
    ) -> PerceptionISPResult:
        """Run all Perception ISP blocks on one RAW frame."""

        metadata, calibration, exposures, provenance = self._sensor_interface(raw)
        normalized_payload = self._raw_physical_normalization(exposures, metadata, calibration)
        hdr_payload = self._hdr_fusion(normalized_payload, metadata, calibration)
        cfa_payload = self._cfa_pixel_structure_decoder(hdr_payload, metadata, calibration)
        noise_payload = self._noise_uncertainty_engine(hdr_payload, cfa_payload, metadata, calibration)
        edge_payload = self._edge_structure_engine(cfa_payload, noise_payload, hdr_payload, calibration)
        color_payload = self._color_spectral_engine(cfa_payload, noise_payload, hdr_payload, calibration)
        geometry_payload = self._optics_geometry_timing_engine(color_payload, metadata, calibration)
        temporal_payload = self._led_flicker_temporal_engine(
            color_payload,
            hdr_payload,
            previous_state,
            metadata,
        )
        formation_payload = self._task_specific_image_formation(
            cfa_payload,
            hdr_payload,
            noise_payload,
            edge_payload,
            color_payload,
            temporal_payload,
            geometry_payload,
            calibration,
        )
        health = self._safety_health_monitor(
            hdr_payload,
            noise_payload,
            edge_payload,
            color_payload,
            metadata,
        )
        runtime = RuntimeController().suggest(health, temporal_payload, self.config)
        metadata_packet = self._metadata_packet(metadata, calibration, geometry_payload, health, runtime, provenance)
        metadata_packet["_accurate_maps"] = {
            "noise_variance": np.asarray(noise_payload["noise_variance"], dtype=np.float64),
            "snr_map": np.asarray(noise_payload["snr_map"], dtype=np.float64),
            "saturation": np.asarray(hdr_payload["saturation"], dtype=np.float64),
            "clipping_distance": np.asarray(hdr_payload["clipping_distance"], dtype=np.float64),
            "hdr_confidence": np.asarray(hdr_payload["hdr_confidence"], dtype=np.float64),
            "edge_strength": np.asarray(edge_payload["edge_strength"], dtype=np.float64),
            "edge_confidence": np.asarray(edge_payload["edge_confidence"], dtype=np.float64),
            "edge_evidence": np.asarray(edge_payload["edge_evidence"], dtype=np.float64),
            "demosaic_confidence": np.asarray(edge_payload["demosaic_confidence"], dtype=np.float64),
            "hdr_exposure_source": np.asarray(hdr_payload["exposure_source"], dtype=np.float64),
            "lens_gain": np.asarray(normalized_payload["lens_gain"], dtype=np.float64),
            "color_confidence": np.asarray(color_payload["color_confidence"], dtype=np.float64),
            "ir_or_clear": np.maximum(
                np.asarray(color_payload["ir_channel"], dtype=np.float64),
                np.asarray(color_payload["clear_channel"], dtype=np.float64),
            ),
            "blur_focus_confidence": np.asarray(edge_payload["blur_focus_confidence"], dtype=np.float64),
            "mtf_confidence": np.asarray(edge_payload["mtf_confidence"], dtype=np.float64),
            "psf_sigma": np.asarray(edge_payload["psf_sigma"], dtype=np.float64),
            "psf_blur_confidence": np.asarray(edge_payload["psf_blur_confidence"], dtype=np.float64),
            "psf_edge_likelihood": np.asarray(edge_payload["psf_edge_likelihood"], dtype=np.float64),
        }
        metadata_packet["_fast_maps"] = {
            "luma": np.asarray(cfa_payload["luma"], dtype=np.float64),
            "edge_strength": np.asarray(edge_payload["edge_strength"], dtype=np.float64),
            "edge_confidence": np.asarray(edge_payload["edge_confidence"], dtype=np.float64),
            "edge_evidence": np.asarray(edge_payload["edge_evidence"], dtype=np.float64),
            "temporal_difference": np.asarray(temporal_payload["temporal_difference"], dtype=np.float64),
            "saturation": np.asarray(hdr_payload["saturation"], dtype=np.float64),
            "noise_variance": np.asarray(noise_payload["noise_variance"], dtype=np.float64),
        }
        accurate = self._format_accurate_path(formation_payload, metadata_packet)
        fast = self._format_fast_path(
            formation_payload,
            hdr_payload,
            noise_payload,
            edge_payload,
            temporal_payload,
            metadata_packet,
        )
        maps = self._collect_maps(
            normalized_payload,
            hdr_payload,
            cfa_payload,
            noise_payload,
            edge_payload,
            color_payload,
            geometry_payload,
            temporal_payload,
        )
        next_state = PreviousFrameState(
            luma=np.asarray(cfa_payload["luma"], dtype=np.float64).copy(),
            rgb=np.asarray(formation_payload["vision_rgb"], dtype=np.float64).copy(),
            timestamp_us=float(metadata.timestamp_us),
            frame_counter=int(metadata.frame_counter),
        )
        public_metadata = {key: value for key, value in metadata_packet.items() if not key.startswith("_")}
        return PerceptionISPResult(
            human_rgb=formation_payload.get("human_rgb"),
            vision_rgb=formation_payload["vision_rgb"],
            raw_normalized=hdr_payload["fused"],
            accurate=accurate,
            fast=fast,
            maps=maps,
            metadata=public_metadata,
            health=health,
            next_state=next_state,
        )

    # Module 0 and 1.
    def _sensor_interface(
        self,
        raw: RawFrame,
    ) -> Tuple[SensorMetadata, CalibrationProfile, ArrayF, Mapping[str, Any]]:
        metadata = raw.metadata
        calibration = raw.calibration or self.default_calibration
        if calibration is None:
            calibration = self.default_calibration
        if metadata.cfa_pattern == "RGGB" and calibration.cfa_pattern:
            metadata = SensorMetadata(**{**metadata_to_dict(metadata), "cfa_pattern": calibration.cfa_pattern})
        exposures = ensure_exposure_first(raw.data)
        if exposures.shape[1] < 2 or exposures.shape[2] < 2:
            raise ValueError("raw frame must contain at least 2x2 pixels")
        return metadata, calibration, exposures, dict(raw.provenance)

    # Module 2.
    def _raw_physical_normalization(
        self,
        exposures: ArrayF,
        metadata: SensorMetadata,
        calibration: CalibrationProfile,
    ) -> Dict[str, Any]:
        rows, cols = exposures.shape[1], exposures.shape[2]
        shape = (rows, cols)
        black = float(calibration.black_level)
        white = max(float(calibration.white_level), black + EPS)
        values = np.maximum(exposures, 0.0)
        if float(calibration.companding_gamma) != 1.0:
            values = np.power(values / white, float(calibration.companding_gamma)) * white

        corrected = (values - black) / (white - black)
        corrected = np.maximum(corrected, 0.0)

        fpn = _optional_map(calibration.fpn_offset, shape, 0.0)
        dsnu = _optional_map(calibration.dsnu_offset, shape, 0.0)
        prnu = np.maximum(_optional_map(calibration.prnu_gain, shape, 1.0), EPS)
        lens_gain = _optional_map(calibration.lens_shading_gain, shape, 1.0)

        analog_gains = _metadata_tuple(metadata.analog_gains, corrected.shape[0], 1.0)
        digital_gains = _metadata_tuple(metadata.digital_gains, corrected.shape[0], 1.0)
        for index in range(corrected.shape[0]):
            gain = max(float(analog_gains[index]) * float(digital_gains[index]), EPS)
            corrected[index] = (corrected[index] - fpn - dsnu) / gain
            corrected[index] = corrected[index] / prnu
            corrected[index] = corrected[index] * lens_gain

        corrected = self._defect_pixel_correction(corrected, calibration.defect_pixels)
        corrected = np.clip(corrected, 0.0, 4.0)
        defect_confidence = np.ones(shape, dtype=np.float64)
        for row, col in calibration.defect_pixels:
            if 0 <= int(row) < rows and 0 <= int(col) < cols:
                defect_confidence[int(row), int(col)] = 0.0
        calibration_residual = np.full(shape, float(calibration.calibration_residual_var), dtype=np.float64)
        return {
            "normalized_exposures": corrected,
            "lens_gain": lens_gain,
            "defect_confidence": defect_confidence,
            "calibration_residual_var": calibration_residual,
            "black_level": black,
            "white_level": white,
        }

    def _defect_pixel_correction(
        self,
        exposures: ArrayF,
        defect_pixels: Sequence[Tuple[int, int]],
    ) -> ArrayF:
        if not defect_pixels:
            return exposures
        corrected = exposures.copy()
        for exposure_index in range(corrected.shape[0]):
            plane = corrected[exposure_index]
            smooth = box_filter(plane, radius=1)
            for row, col in defect_pixels:
                r, c = int(row), int(col)
                if 0 <= r < plane.shape[0] and 0 <= c < plane.shape[1]:
                    plane[r, c] = smooth[r, c]
        return corrected

    # Module 5.
    def _hdr_fusion(
        self,
        normalized_payload: Mapping[str, Any],
        metadata: SensorMetadata,
        calibration: CalibrationProfile,
    ) -> Dict[str, Any]:
        exposures = np.asarray(normalized_payload["normalized_exposures"], dtype=np.float64)
        count, rows, cols = exposures.shape
        exposure_times = np.asarray(_metadata_tuple(metadata.exposure_times_us, count, 1.0), dtype=np.float64)
        analog_gains = np.asarray(_metadata_tuple(metadata.analog_gains, count, 1.0), dtype=np.float64)
        digital_gains = np.asarray(_metadata_tuple(metadata.digital_gains, count, 1.0), dtype=np.float64)
        scale = np.maximum(exposure_times * analog_gains * digital_gains, EPS)
        scale = scale / max(float(np.max(scale)), EPS)
        radiance = exposures / scale[:, None, None]
        saturation = exposures >= float(self.config.hdr_saturation_threshold)
        low_signal = exposures <= float(self.config.hdr_low_signal_threshold)
        shot_var = float(calibration.shot_noise_coeff) * np.maximum(radiance, 0.0)
        read_var = float(calibration.read_noise_var)
        snr = radiance / np.sqrt(shot_var + read_var + EPS)
        snr_weight = snr / (snr + 1.0)
        nonsat_weight = np.where(saturation, 0.0, 1.0)
        low_weight = np.where(low_signal, 0.35, 1.0)
        if count > 1:
            median = np.median(radiance, axis=0)
            motion_delta = np.abs(radiance - median[None, :, :])
            motion_weight = np.exp(-motion_delta / (0.10 + np.abs(median)[None, :, :]))
            ghost_map = np.std(radiance, axis=0) / (np.mean(np.abs(radiance), axis=0) + EPS)
        else:
            motion_weight = np.ones_like(radiance)
            ghost_map = np.zeros((rows, cols), dtype=np.float64)
        weights = np.maximum(snr_weight * nonsat_weight * low_weight * motion_weight, 0.0)
        fallback_index = np.argmin(exposures, axis=0)
        fallback = np.take_along_axis(radiance, fallback_index[None, :, :], axis=0)[0]
        weight_sum = np.sum(weights, axis=0)
        fused = np.where(
            weight_sum > EPS,
            np.sum(weights * radiance, axis=0) / np.maximum(weight_sum, EPS),
            fallback,
        )
        source = np.argmax(weights, axis=0).astype(np.float64)
        source = np.where(weight_sum > EPS, source, fallback_index.astype(np.float64))
        fused = np.clip(fused, 0.0, 4.0)
        any_saturation = np.any(saturation, axis=0).astype(np.float64)
        clipping_distance = np.clip(
            (float(self.config.hdr_saturation_threshold) - np.max(exposures, axis=0)) / max(float(self.config.hdr_saturation_threshold), EPS),
            0.0,
            1.0,
        )
        confidence = np.clip(weight_sum / max(float(count), 1.0), 0.0, 1.0)
        led_flicker_confidence = np.clip(ghost_map * any_saturation, 0.0, 1.0)
        return {
            "fused": fused,
            "radiance_exposures": radiance,
            "weights": weights,
            "exposure_source": source,
            "saturation": any_saturation,
            "clipping_distance": clipping_distance,
            "per_exposure_saturation": saturation.astype(np.float64),
            "hdr_confidence": confidence,
            "ghost_motion_artifact": np.clip(ghost_map, 0.0, 1.0),
            "led_flicker_confidence_hdr": led_flicker_confidence,
            "lens_gain": normalized_payload["lens_gain"],
            "defect_confidence": normalized_payload["defect_confidence"],
            "calibration_residual_var": normalized_payload["calibration_residual_var"],
        }

    # Module 4.
    def _cfa_pixel_structure_decoder(
        self,
        hdr_payload: Mapping[str, Any],
        metadata: SensorMetadata,
        calibration: CalibrationProfile,
    ) -> Dict[str, Any]:
        fused = np.asarray(hdr_payload["fused"], dtype=np.float64)
        pattern = str(metadata.cfa_pattern or calibration.cfa_pattern or "RGGB").upper().replace("-", "")
        masks = _cfa_masks(pattern, fused.shape)
        if pattern in {"RGGB", "BGGR", "GRBG", "GBRG"}:
            decoded = _decode_bayer(fused, masks, method=self.config.demosaic_method)
        elif pattern in {"RCCB", "RCCC", "RCCG"}:
            decoded = _decode_clear_cfa(fused, masks, pattern)
        elif pattern in {"RGBIR", "RGBIR2X2"}:
            decoded = _decode_rgb_ir(fused, masks, calibration)
        elif pattern in {"MONO", "MONOCHROME", "THERMAL"}:
            decoded = _decode_monochrome(fused)
        else:
            decoded = _decode_bayer(fused, _cfa_masks("RGGB", fused.shape), method=self.config.demosaic_method)
            decoded["cfa_warning"] = "unsupported pattern treated as RGGB"
        decoded["pattern"] = pattern
        decoded["masks"] = masks
        return decoded

    # Module 3.
    def _noise_uncertainty_engine(
        self,
        hdr_payload: Mapping[str, Any],
        cfa_payload: Mapping[str, Any],
        metadata: SensorMetadata,
        calibration: CalibrationProfile,
    ) -> Dict[str, Any]:
        signal = np.maximum(np.asarray(hdr_payload["fused"], dtype=np.float64), 0.0)
        exposure = max(float(_metadata_tuple(metadata.exposure_times_us, 1, 1.0)[0]), EPS)
        temp_delta = max(float(metadata.temperature_c) - 25.0, 0.0)
        raw_var = (
            float(calibration.shot_noise_coeff) * signal
            + float(calibration.read_noise_var)
            + float(calibration.dark_current_coeff) * temp_delta * exposure
            + float(calibration.quantization_var)
            + np.asarray(hdr_payload["calibration_residual_var"], dtype=np.float64)
        )
        lens_gain = np.asarray(hdr_payload["lens_gain"], dtype=np.float64)
        variance = np.maximum(raw_var * lens_gain * lens_gain, EPS)
        snr = signal / np.sqrt(variance + EPS)
        snr_map = snr / (snr + 1.0)
        gx, gy = gradient_xy(np.asarray(cfa_payload["luma"], dtype=np.float64))
        gradient = np.sqrt(gx * gx + gy * gy)
        noise_normalized_gradient = gradient / np.sqrt(2.0 * variance + EPS)
        temporal_noise_confidence = np.clip(snr_map * np.asarray(hdr_payload["hdr_confidence"], dtype=np.float64), 0.0, 1.0)
        return {
            "noise_variance": variance,
            "snr": snr,
            "snr_map": np.clip(snr_map, 0.0, 1.0),
            "noise_normalized_gradient": noise_normalized_gradient,
            "temporal_noise_confidence": temporal_noise_confidence,
            "calibration_residual": np.asarray(hdr_payload["calibration_residual_var"], dtype=np.float64),
        }

    # Module 6.
    def _edge_structure_engine(
        self,
        cfa_payload: Mapping[str, Any],
        noise_payload: Mapping[str, Any],
        hdr_payload: Mapping[str, Any],
        calibration: CalibrationProfile,
    ) -> Dict[str, Any]:
        luma = np.asarray(cfa_payload["luma"], dtype=np.float64)
        rg = np.asarray(cfa_payload.get("r_minus_g", np.zeros_like(luma)), dtype=np.float64)
        bg = np.asarray(cfa_payload.get("b_minus_g", np.zeros_like(luma)), dtype=np.float64)
        confidence_base = np.asarray(noise_payload["snr_map"], dtype=np.float64)
        variance = np.asarray(noise_payload["noise_variance"], dtype=np.float64)
        channels = (luma, rg, bg)
        weights = (
            confidence_base / np.maximum(variance, EPS),
            0.5 * confidence_base / np.maximum(variance, EPS),
            0.5 * confidence_base / np.maximum(variance, EPS),
        )
        jxx = np.zeros_like(luma)
        jxy = np.zeros_like(luma)
        jyy = np.zeros_like(luma)
        gradients: List[Tuple[ArrayF, ArrayF]] = []
        for channel, weight in zip(channels, weights):
            gx, gy = gradient_xy(channel)
            gradients.append((gx, gy))
            jxx += weight * gx * gx
            jxy += weight * gx * gy
            jyy += weight * gy * gy
        trace = jxx + jyy
        delta = np.sqrt(np.maximum((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy, 0.0))
        lambda1 = 0.5 * (trace + delta)
        lambda2 = 0.5 * (trace - delta)
        edge_strength = np.sqrt(np.maximum(lambda1, 0.0))
        if float(np.max(edge_strength)) > EPS:
            edge_strength = edge_strength / float(np.max(edge_strength))
        mtf = _optional_map(calibration.mtf_confidence_map, luma.shape, 1.0)
        psf_sigma = np.maximum(_optional_map(calibration.psf_sigma_map, luma.shape, 0.0), 0.0)
        psf_blur_confidence = _psf_blur_confidence(psf_sigma)
        structure_edge_confidence = (lambda1 - lambda2) / (lambda1 + lambda2 + EPS)
        saturation_gate = 1.0 - np.asarray(hdr_payload["saturation"], dtype=np.float64)
        edge_confidence = np.clip(structure_edge_confidence * mtf * psf_blur_confidence * saturation_gate * confidence_base, 0.0, 1.0)
        edge_evidence = _combined_edge_evidence(edge_confidence, edge_strength)
        orientation = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + EPS)

        luma_edge = np.sqrt(gradients[0][0] * gradients[0][0] + gradients[0][1] * gradients[0][1])
        chroma_r = np.sqrt(gradients[1][0] * gradients[1][0] + gradients[1][1] * gradients[1][1])
        chroma_b = np.sqrt(gradients[2][0] * gradients[2][0] + gradients[2][1] * gradients[2][1])
        chroma_edge = np.maximum(chroma_r, chroma_b)
        luma_norm = luma_edge / max(float(np.max(luma_edge)), EPS)
        chroma_norm = chroma_edge / max(float(np.max(chroma_edge)), EPS)
        edge_type = np.zeros_like(luma)
        edge_type[(luma_norm > 0.12) & (chroma_norm <= 0.12)] = 1.0
        edge_type[(luma_norm <= 0.12) & (chroma_norm > 0.12)] = 2.0
        edge_type[(luma_norm > 0.12) & (chroma_norm > 0.12)] = 3.0

        blur_focus_confidence = _blur_focus_confidence(edge_strength, edge_confidence, self.config.blur_edge_percentile)
        demosaic_confidence = np.asarray(cfa_payload.get("demosaic_confidence", np.ones_like(luma)), dtype=np.float64)
        demosaic_confidence = np.clip(demosaic_confidence * edge_confidence + (1.0 - edge_strength) * 0.5, 0.0, 1.0)
        psf_likelihood = np.clip(edge_confidence - np.asarray(hdr_payload["ghost_motion_artifact"], dtype=np.float64), 0.0, 1.0)
        return {
            "edge_strength": edge_strength,
            "edge_confidence": edge_confidence,
            "edge_evidence": edge_evidence,
            "edge_orientation": orientation,
            "luma_edge": np.clip(luma_norm, 0.0, 1.0),
            "chroma_edge": np.clip(chroma_norm, 0.0, 1.0),
            "edge_type": edge_type,
            "demosaic_confidence": demosaic_confidence,
            "blur_focus_confidence": blur_focus_confidence,
            "mtf_confidence": mtf,
            "psf_sigma": psf_sigma,
            "psf_blur_confidence": psf_blur_confidence,
            "psf_edge_likelihood": psf_likelihood,
        }

    # Module 7.
    def _color_spectral_engine(
        self,
        cfa_payload: Mapping[str, Any],
        noise_payload: Mapping[str, Any],
        hdr_payload: Mapping[str, Any],
        calibration: CalibrationProfile,
    ) -> Dict[str, Any]:
        rgb = np.clip(np.asarray(cfa_payload["rgb"], dtype=np.float64), 0.0, 4.0)
        camera_rgb = np.clip(apply_matrix_rgb(rgb, calibration.color_matrix), 0.0, 4.0)
        perception_rgb = np.clip(apply_matrix_rgb(rgb, calibration.perception_color_matrix), 0.0, 4.0)
        g = np.maximum(camera_rgb[:, :, 1], EPS)
        log_rg = safe_log_ratio(camera_rgb[:, :, 0], g)
        log_bg = safe_log_ratio(camera_rgb[:, :, 2], g)
        r_minus_g = camera_rgb[:, :, 0] - camera_rgb[:, :, 1]
        b_minus_g = camera_rgb[:, :, 2] - camera_rgb[:, :, 1]
        means = np.maximum(np.mean(camera_rgb[:, :, :3], axis=(0, 1)), EPS)
        wb_gains = float(np.mean(means)) / means
        wb_gains = wb_gains / max(float(wb_gains[1]), EPS)
        imbalance = float((np.max(means) - np.min(means)) / max(float(np.mean(means)), EPS))
        wb_confidence = float(np.clip(1.0 - imbalance, 0.0, 1.0))
        color_confidence = np.clip(
            np.asarray(noise_payload["snr_map"], dtype=np.float64)
            * (1.0 - np.asarray(hdr_payload["saturation"], dtype=np.float64))
            * wb_confidence
            * np.asarray(hdr_payload["hdr_confidence"], dtype=np.float64),
            0.0,
            1.0,
        )
        ir_channel = np.asarray(cfa_payload.get("ir_channel", np.zeros(camera_rgb.shape[:2])), dtype=np.float64)
        ir_contamination = np.clip(ir_channel / (np.mean(camera_rgb, axis=2) + ir_channel + EPS), 0.0, 1.0)
        return {
            "camera_rgb": camera_rgb,
            "perception_rgb": perception_rgb,
            "log_r_over_g": log_rg,
            "log_b_over_g": log_bg,
            "r_minus_g": r_minus_g,
            "b_minus_g": b_minus_g,
            "color_confidence": color_confidence,
            "illuminant_estimate_rgb": means,
            "wb_gains_rgb": wb_gains,
            "wb_confidence": wb_confidence,
            "ir_contamination": ir_contamination,
            "ir_channel": ir_channel,
            "clear_channel": np.asarray(cfa_payload.get("clear_channel", np.zeros(camera_rgb.shape[:2])), dtype=np.float64),
        }

    # Module 8.
    def _optics_geometry_timing_engine(
        self,
        color_payload: Mapping[str, Any],
        metadata: SensorMetadata,
        calibration: CalibrationProfile,
    ) -> Dict[str, Any]:
        rgb = np.asarray(color_payload["perception_rgb"], dtype=np.float64)
        if self.config.accurate_enable_dewarp:
            accurate_rgb = nearest_dewarp(rgb, calibration.distortion_coeffs)
            transform = "nearest_radial_distortion_correction"
        else:
            accurate_rgb = rgb.copy()
            transform = "identity"
        rows, cols = rgb.shape[:2]
        row_times = row_timestamp_map(rows, metadata.timestamp_us, metadata.line_time_us)
        rolling_row_fraction = np.linspace(0.0, 1.0, rows, dtype=np.float64)
        return {
            "accurate_rgb_geometry": accurate_rgb,
            "row_timestamps_us": row_times,
            "rolling_row_fraction": rolling_row_fraction,
            "intrinsic_matrix": calibration.intrinsic_matrix.copy(),
            "distortion_coeffs": tuple(float(v) for v in calibration.distortion_coeffs),
            "extrinsic_matrix": calibration.extrinsic_matrix.copy(),
            "roi_coordinate_transform": transform,
            "camera_sync_group": metadata.camera_id,
        }

    # Module 9.
    def _led_flicker_temporal_engine(
        self,
        color_payload: Mapping[str, Any],
        hdr_payload: Mapping[str, Any],
        previous_state: Optional[PreviousFrameState],
        metadata: SensorMetadata,
    ) -> Dict[str, Any]:
        rgb = np.asarray(color_payload["perception_rgb"], dtype=np.float64)
        luma = np.mean(rgb[:, :, :3], axis=2)
        if previous_state is None or previous_state.luma is None or previous_state.luma.shape != luma.shape:
            temporal_difference = np.zeros_like(luma)
            temporal_consistency = np.ones_like(luma)
        else:
            prev = np.asarray(previous_state.luma, dtype=np.float64)
            temporal_difference = np.abs(luma - prev)
            temporal_consistency = np.exp(-temporal_difference / max(float(self.config.temporal_flicker_threshold), EPS))
        bright_light = np.clip((luma - 0.65) / 0.35, 0.0, 1.0)
        hdr_flicker = np.asarray(hdr_payload["led_flicker_confidence_hdr"], dtype=np.float64)
        flicker = np.clip((1.0 - temporal_consistency) * bright_light + hdr_flicker, 0.0, 1.0)
        led_state = np.where(flicker > 0.5, 1.0, 0.0)
        return {
            "temporal_difference": temporal_difference,
            "temporal_consistency": temporal_consistency,
            "led_flicker_confidence": flicker,
            "light_source_confidence": np.clip(bright_light * (1.0 - flicker * 0.5), 0.0, 1.0),
            "led_state_track": led_state,
            "previous_frame_counter": -1 if previous_state is None else int(previous_state.frame_counter),
            "frame_delta_us": 0.0 if previous_state is None else float(metadata.timestamp_us - previous_state.timestamp_us),
        }

    # Module 10.
    def _task_specific_image_formation(
        self,
        cfa_payload: Mapping[str, Any],
        hdr_payload: Mapping[str, Any],
        noise_payload: Mapping[str, Any],
        edge_payload: Mapping[str, Any],
        color_payload: Mapping[str, Any],
        temporal_payload: Mapping[str, Any],
        geometry_payload: Mapping[str, Any],
        calibration: CalibrationProfile,
    ) -> Dict[str, Any]:
        edge_confidence = np.asarray(edge_payload["edge_confidence"], dtype=np.float64)
        demosaic_confidence = np.asarray(edge_payload["demosaic_confidence"], dtype=np.float64)
        perception_rgb = _suppress_demosaic_artifacts(
            np.asarray(color_payload["perception_rgb"], dtype=np.float64),
            edge_confidence,
            demosaic_confidence,
            self.config.demosaic_artifact_suppression,
        )
        weak_denoised = edge_aware_denoise(perception_rgb, edge_confidence, self.config.denoise_strength)
        vision_rgb = _tone_map_rgb(weak_denoised, self.config.tone_mapping, self.config.gamma)
        human_rgb = None
        if self.config.include_human_view:
            human_source = _suppress_demosaic_artifacts(
                np.asarray(color_payload["camera_rgb"], dtype=np.float64),
                edge_confidence,
                demosaic_confidence,
                self.config.demosaic_artifact_suppression,
            )
            human_base = edge_aware_denoise(human_source, edge_confidence, 0.45)
            human_mode = "human_log" if self.config.tone_mapping.lower() == "log" else self.config.tone_mapping
            human_rgb = _tone_map_rgb(human_base, human_mode, self.config.gamma)

        raw_like_channels = [
            np.asarray(cfa_payload["luma"], dtype=np.float64),
            np.asarray(cfa_payload.get("r_minus_g", np.zeros_like(edge_confidence)), dtype=np.float64),
            np.asarray(cfa_payload.get("b_minus_g", np.zeros_like(edge_confidence)), dtype=np.float64),
            np.asarray(noise_payload["snr_map"], dtype=np.float64),
            np.asarray(edge_payload["edge_strength"], dtype=np.float64),
        ]
        raw_like = np.stack(raw_like_channels, axis=2)

        accurate_rgb = _suppress_demosaic_artifacts(
            np.asarray(geometry_payload["accurate_rgb_geometry"], dtype=np.float64),
            edge_confidence,
            demosaic_confidence,
            self.config.demosaic_artifact_suppression,
        )
        accurate_rgb = _tone_map_rgb(edge_aware_denoise(accurate_rgb, edge_confidence, self.config.denoise_strength), self.config.tone_mapping, self.config.gamma)
        return {
            "vision_rgb": np.clip(vision_rgb, 0.0, 1.0),
            "human_rgb": None if human_rgb is None else np.clip(human_rgb, 0.0, 1.0),
            "raw_like": raw_like,
            "accurate_rgb": np.clip(accurate_rgb, 0.0, 1.0),
        }

    # Module 11 plus safety.
    def _format_accurate_path(
        self,
        formation_payload: Mapping[str, Any],
        metadata_packet: Mapping[str, Any],
    ) -> AccuratePathOutput:
        channels = [
            ("rgb_r", formation_payload["accurate_rgb"][:, :, 0]),
            ("rgb_g", formation_payload["accurate_rgb"][:, :, 1]),
            ("rgb_b", formation_payload["accurate_rgb"][:, :, 2]),
        ]
        for name, array in metadata_packet["_accurate_maps"].items():
            channels.append((name, array))
        tensor = np.stack([np.asarray(array, dtype=np.float64) for _, array in channels], axis=2)
        public_metadata = {key: value for key, value in metadata_packet.items() if not key.startswith("_")}
        return AccuratePathOutput(
            tensor=tensor,
            channels=tuple(name for name, _ in channels),
            metadata=public_metadata,
        )

    def _format_fast_path(
        self,
        formation_payload: Mapping[str, Any],
        hdr_payload: Mapping[str, Any],
        noise_payload: Mapping[str, Any],
        edge_payload: Mapping[str, Any],
        temporal_payload: Mapping[str, Any],
        metadata_packet: Mapping[str, Any],
    ) -> FastPathOutput:
        maps = metadata_packet["_fast_maps"]
        roi = _fast_roi(maps["luma"].shape, self.config.fast_path_roi, self.config.fast_path_fraction)
        row0, col0, row1, col1 = roi
        stripe_height = min(max(int(self.config.fast_path_stripe_height), 1), row1 - row0)
        stripe_row0 = row1 - stripe_height if self.config.fast_path_roi == "bottom" else row0
        stripe = (stripe_row0, col0, row1 if self.config.fast_path_roi == "bottom" else row0 + stripe_height, col1)
        sr0, sc0, sr1, sc1 = stripe
        fast_channels = [
            ("luma", maps["luma"][sr0:sr1, sc0:sc1]),
            ("edge_strength", maps["edge_strength"][sr0:sr1, sc0:sc1]),
            ("edge_confidence", maps["edge_confidence"][sr0:sr1, sc0:sc1]),
            ("temporal_difference", maps["temporal_difference"][sr0:sr1, sc0:sc1]),
            ("saturation", maps["saturation"][sr0:sr1, sc0:sc1]),
            ("noise_variance", maps["noise_variance"][sr0:sr1, sc0:sc1]),
        ]
        tensor = np.stack([array for _, array in fast_channels], axis=2)
        tensor = block_reduce_mean(tensor, self.config.fast_path_low_res_factor)
        edge_packets = self._edge_packets(
            edge_payload,
            noise_payload,
            hdr_payload,
            temporal_payload,
            stripe,
        )
        line_time = float(metadata_packet["frame"]["line_time_us"])
        estimated_latency_us = (
            (stripe_height * line_time)
            + float(metadata_packet["latency_model"]["streaming_isp_delay_us"])
            + float(metadata_packet["latency_model"]["early_model_budget_us"])
        )
        public_metadata = {key: value for key, value in metadata_packet.items() if not key.startswith("_")}
        return FastPathOutput(
            tensor=tensor,
            channels=tuple(name for name, _ in fast_channels),
            roi=stripe,
            edge_packets=tuple(edge_packets),
            estimated_latency_us=float(estimated_latency_us),
            metadata=public_metadata,
        )

    def _edge_packets(
        self,
        edge_payload: Mapping[str, Any],
        noise_payload: Mapping[str, Any],
        hdr_payload: Mapping[str, Any],
        temporal_payload: Mapping[str, Any],
        roi: Tuple[int, int, int, int],
    ) -> List[EdgePacket]:
        row0, col0, row1, col1 = roi
        strength = np.asarray(edge_payload["edge_strength"], dtype=np.float64)[row0:row1, col0:col1]
        confidence = np.asarray(edge_payload["edge_confidence"], dtype=np.float64)[row0:row1, col0:col1]
        score = strength * confidence
        mask = score >= float(self.config.edge_packet_threshold)
        if not bool(np.any(mask)):
            return []
        flat_score = np.where(mask, score, -1.0).reshape(-1)
        count = min(int(self.config.max_edge_packets), flat_score.size)
        if count <= 0:
            return []
        indices = np.argpartition(flat_score, -count)[-count:]
        indices = indices[np.argsort(flat_score[indices])[::-1]]
        orientation = np.asarray(edge_payload["edge_orientation"], dtype=np.float64)
        snr_map = np.asarray(noise_payload["snr_map"], dtype=np.float64)
        source = np.asarray(hdr_payload["exposure_source"], dtype=np.float64)
        saturation = np.asarray(hdr_payload["saturation"], dtype=np.float64)
        temporal_consistency = np.asarray(temporal_payload["temporal_consistency"], dtype=np.float64)
        packets: List[EdgePacket] = []
        width = strength.shape[1]
        for flat in indices:
            if flat_score[flat] < float(self.config.edge_packet_threshold):
                continue
            local_r = int(flat // width)
            local_c = int(flat % width)
            r, c = row0 + local_r, col0 + local_c
            packets.append(
                EdgePacket(
                    row=r,
                    col=c,
                    edge_strength=float(edge_payload["edge_strength"][r, c]),
                    edge_orientation_rad=float(orientation[r, c]),
                    edge_confidence=float(confidence[local_r, local_c]),
                    noise_confidence=float(snr_map[r, c]),
                    hdr_source=int(source[r, c]),
                    saturation_state=float(saturation[r, c]),
                    motion_consistency=float(temporal_consistency[r, c]),
                )
            )
        return packets

    def _safety_health_monitor(
        self,
        hdr_payload: Mapping[str, Any],
        noise_payload: Mapping[str, Any],
        edge_payload: Mapping[str, Any],
        color_payload: Mapping[str, Any],
        metadata: SensorMetadata,
    ) -> Dict[str, Any]:
        saturation = np.asarray(hdr_payload["saturation"], dtype=np.float64)
        luma = np.mean(np.asarray(color_payload["perception_rgb"], dtype=np.float64), axis=2)
        edge_conf = np.asarray(edge_payload["edge_confidence"], dtype=np.float64)
        blur_conf = np.asarray(edge_payload["blur_focus_confidence"], dtype=np.float64)
        color_conf = np.asarray(color_payload["color_confidence"], dtype=np.float64)
        snr = np.asarray(noise_payload["snr_map"], dtype=np.float64)
        over = float(np.mean(saturation))
        under = float(np.mean(luma < 0.02))
        visibility = float(np.clip(np.mean(snr) * 0.45 + np.mean(edge_conf) * 0.25 + np.mean(color_conf) * 0.30, 0.0, 1.0))
        focus_score = float(np.mean(blur_conf))
        tint_score = float(color_payload["wb_confidence"])
        frozen_frame_suspect = False
        camera_health = "ok"
        warnings: List[str] = []
        if over > 0.10:
            warnings.append("over_exposure")
        if under > 0.75:
            warnings.append("under_exposure")
        if visibility < 0.20:
            warnings.append("low_visibility")
        if focus_score < 0.20:
            warnings.append("blur_or_defocus")
        if tint_score < 0.30:
            warnings.append("color_tint")
        if warnings:
            camera_health = "degraded"
        return {
            "camera_health_status": camera_health,
            "warnings": warnings,
            "visibility_confidence": visibility,
            "sensor_validity": float(np.clip(1.0 - max(over, under), 0.0, 1.0)),
            "dnn_input_validity": float(np.clip(0.5 * visibility + 0.5 * np.mean(hdr_payload["hdr_confidence"]), 0.0, 1.0)),
            "over_exposure_fraction": over,
            "under_exposure_fraction": under,
            "focus_confidence": focus_score,
            "color_tint_confidence": tint_score,
            "frozen_frame_suspect": frozen_frame_suspect,
            "metadata_consistency": {
                "frame_counter": int(metadata.frame_counter),
                "line_time_us": float(metadata.line_time_us),
                "rolling_shutter_time_us": float(metadata.rolling_shutter_time_us),
            },
        }

    def _metadata_packet(
        self,
        metadata: SensorMetadata,
        calibration: CalibrationProfile,
        geometry_payload: Mapping[str, Any],
        health: Mapping[str, Any],
        runtime: RuntimeControlSuggestion,
        raw_provenance: Mapping[str, Any],
    ) -> Dict[str, Any]:
        frame = metadata_to_dict(metadata)
        row_times = np.asarray(geometry_payload["row_timestamps_us"], dtype=np.float64)
        packet: Dict[str, Any] = {
            "frame": frame,
            "calibration": {
                "cfa_pattern": calibration.cfa_pattern,
                "black_level": float(calibration.black_level),
                "white_level": float(calibration.white_level),
                "lens_profile_id": metadata.lens_profile_id,
                "color_profile_id": metadata.color_profile_id,
                "noise_model_id": metadata.noise_model_id,
            },
            "processing": {
                "demosaic_method": str(self.config.demosaic_method),
                "demosaic_artifact_suppression": float(self.config.demosaic_artifact_suppression),
                "tone_mapping": str(self.config.tone_mapping),
                "denoise_strength": float(self.config.denoise_strength),
            },
            "geometry": {
                "intrinsic_matrix": np.asarray(geometry_payload["intrinsic_matrix"], dtype=np.float64).tolist(),
                "distortion_coeffs": list(geometry_payload["distortion_coeffs"]),
                "extrinsic_matrix": np.asarray(geometry_payload["extrinsic_matrix"], dtype=np.float64).tolist(),
                "row_timestamp_start_us": float(row_times[0]) if row_times.size else float(metadata.timestamp_us),
                "row_timestamp_end_us": float(row_times[-1]) if row_times.size else float(metadata.timestamp_us),
                "readout_direction": metadata.readout_direction,
                "roi_coordinate_transform": str(geometry_payload["roi_coordinate_transform"]),
                "camera_sync_group": str(geometry_payload["camera_sync_group"]),
            },
            "health": health,
            "raw_provenance": dict(raw_provenance),
            "runtime_control_suggestion": runtime.to_dict(),
            "latency_model": {
                "streaming_isp_delay_us": 900.0,
                "early_model_budget_us": 2500.0,
                "accurate_model_budget_us": 15000.0,
            },
        }
        return packet

    def _collect_maps(
        self,
        normalized_payload: Mapping[str, Any],
        hdr_payload: Mapping[str, Any],
        cfa_payload: Mapping[str, Any],
        noise_payload: Mapping[str, Any],
        edge_payload: Mapping[str, Any],
        color_payload: Mapping[str, Any],
        geometry_payload: Mapping[str, Any],
        temporal_payload: Mapping[str, Any],
    ) -> Dict[str, ArrayF]:
        maps: Dict[str, ArrayF] = {
            "lens_gain": np.asarray(normalized_payload["lens_gain"], dtype=np.float64),
            "defect_confidence": np.asarray(normalized_payload["defect_confidence"], dtype=np.float64),
            "hdr_exposure_source": np.asarray(hdr_payload["exposure_source"], dtype=np.float64),
            "saturation": np.asarray(hdr_payload["saturation"], dtype=np.float64),
            "clipping_distance": np.asarray(hdr_payload["clipping_distance"], dtype=np.float64),
            "hdr_confidence": np.asarray(hdr_payload["hdr_confidence"], dtype=np.float64),
            "ghost_motion_artifact": np.asarray(hdr_payload["ghost_motion_artifact"], dtype=np.float64),
            "luma": np.asarray(cfa_payload["luma"], dtype=np.float64),
            "clear_channel": np.asarray(cfa_payload.get("clear_channel", np.zeros_like(hdr_payload["fused"])), dtype=np.float64),
            "ir_channel": np.asarray(cfa_payload.get("ir_channel", np.zeros_like(hdr_payload["fused"])), dtype=np.float64),
            "noise_variance": np.asarray(noise_payload["noise_variance"], dtype=np.float64),
            "snr_map": np.asarray(noise_payload["snr_map"], dtype=np.float64),
            "noise_normalized_gradient": np.asarray(noise_payload["noise_normalized_gradient"], dtype=np.float64),
            "edge_strength": np.asarray(edge_payload["edge_strength"], dtype=np.float64),
            "edge_orientation": np.asarray(edge_payload["edge_orientation"], dtype=np.float64),
            "edge_confidence": np.asarray(edge_payload["edge_confidence"], dtype=np.float64),
            "edge_evidence": np.asarray(edge_payload["edge_evidence"], dtype=np.float64),
            "edge_type": np.asarray(edge_payload["edge_type"], dtype=np.float64),
            "demosaic_confidence": np.asarray(edge_payload["demosaic_confidence"], dtype=np.float64),
            "blur_focus_confidence": np.asarray(edge_payload["blur_focus_confidence"], dtype=np.float64),
            "mtf_confidence": np.asarray(edge_payload["mtf_confidence"], dtype=np.float64),
            "psf_sigma": np.asarray(edge_payload["psf_sigma"], dtype=np.float64),
            "psf_blur_confidence": np.asarray(edge_payload["psf_blur_confidence"], dtype=np.float64),
            "psf_edge_likelihood": np.asarray(edge_payload["psf_edge_likelihood"], dtype=np.float64),
            "color_confidence": np.asarray(color_payload["color_confidence"], dtype=np.float64),
            "log_r_over_g": np.asarray(color_payload["log_r_over_g"], dtype=np.float64),
            "log_b_over_g": np.asarray(color_payload["log_b_over_g"], dtype=np.float64),
            "ir_contamination": np.asarray(color_payload["ir_contamination"], dtype=np.float64),
            "rolling_row_fraction": np.asarray(geometry_payload["rolling_row_fraction"], dtype=np.float64),
            "temporal_difference": np.asarray(temporal_payload["temporal_difference"], dtype=np.float64),
            "temporal_consistency": np.asarray(temporal_payload["temporal_consistency"], dtype=np.float64),
            "led_flicker_confidence": np.asarray(temporal_payload["led_flicker_confidence"], dtype=np.float64),
            "light_source_confidence": np.asarray(temporal_payload["light_source_confidence"], dtype=np.float64),
        }
        return maps


class RuntimeController:
    """Simple rule-based profile controller for the software reference."""

    def suggest(
        self,
        health: Mapping[str, Any],
        temporal_payload: Mapping[str, Any],
        config: PerceptionISPConfig,
    ) -> RuntimeControlSuggestion:
        warnings = set(str(item) for item in health.get("warnings", []))
        flicker = float(np.mean(np.asarray(temporal_payload["led_flicker_confidence"], dtype=np.float64)))
        fast_priority = 0.5
        hdr_priority = 0.5
        denoise = float(config.denoise_strength)
        exposure_priority = "balanced"
        notes: List[str] = []
        if "over_exposure" in warnings or flicker > 0.05:
            hdr_priority = 0.9
            exposure_priority = "highlight_protection"
            notes.append("prioritize HDR/saturation maps for glare or LED scenes")
        if "under_exposure" in warnings or "low_visibility" in warnings:
            denoise = min(0.45, denoise + 0.15)
            exposure_priority = "low_light_snr"
            notes.append("increase noise-map weight and edge-gated denoise")
        if float(health.get("dnn_input_validity", 1.0)) < 0.45:
            fast_priority = 0.85
            notes.append("raise fast-path priority because full-frame confidence is degraded")
        return RuntimeControlSuggestion(
            exposure_priority=exposure_priority,
            hdr_priority=hdr_priority,
            denoise_strength=denoise,
            fast_path_priority=fast_priority,
            enable_dewarp_fast_path=False,
            notes=tuple(notes),
        )


def _metadata_tuple(values: Sequence[float], count: int, default: float) -> Tuple[float, ...]:
    if not values:
        return tuple(float(default) for _ in range(count))
    result = list(float(v) for v in values)
    while len(result) < count:
        result.append(result[-1] if result else float(default))
    return tuple(result[:count])


def _optional_map(values: Optional[ArrayF], shape: Tuple[int, int], default: float) -> ArrayF:
    if values is None:
        return np.full(shape, float(default), dtype=np.float64)
    return resize_nearest(values, shape)


def _cfa_masks(pattern: str, shape: Tuple[int, int]) -> Dict[str, ArrayF]:
    rows, cols = int(shape[0]), int(shape[1])
    pattern = pattern.upper().replace("-", "")
    masks: Dict[str, ArrayF] = {
        "R": np.zeros((rows, cols), dtype=np.float64),
        "G": np.zeros((rows, cols), dtype=np.float64),
        "B": np.zeros((rows, cols), dtype=np.float64),
        "C": np.zeros((rows, cols), dtype=np.float64),
        "IR": np.zeros((rows, cols), dtype=np.float64),
    }
    if pattern == "RGGB":
        tile = (("R", "G"), ("G", "B"))
    elif pattern == "BGGR":
        tile = (("B", "G"), ("G", "R"))
    elif pattern == "GRBG":
        tile = (("G", "R"), ("B", "G"))
    elif pattern == "GBRG":
        tile = (("G", "B"), ("R", "G"))
    elif pattern == "RCCB":
        tile = (("R", "C"), ("C", "B"))
    elif pattern == "RCCC":
        tile = (("R", "C"), ("C", "C"))
    elif pattern == "RCCG":
        tile = (("R", "C"), ("C", "G"))
    elif pattern in {"RGBIR", "RGBIR2X2"}:
        tile = (("R", "G"), ("B", "IR"))
    else:
        tile = (("R", "G"), ("G", "B"))
    for r in range(2):
        for c in range(2):
            masks[tile[r][c]][r::2, c::2] = 1.0
    return masks


def _decode_bayer(fused: ArrayF, masks: Mapping[str, ArrayF], *, method: str = "edge_aware") -> Dict[str, Any]:
    normalized = str(method or "edge_aware").lower().replace("-", "_")
    if normalized in {"bilinear", "linear", "box"}:
        return _decode_bayer_bilinear(fused, masks, method="bilinear")
    return _decode_bayer_edge_aware(fused, masks, method="edge_aware")


def _decode_bayer_bilinear(fused: ArrayF, masks: Mapping[str, ArrayF], *, method: str) -> Dict[str, Any]:
    r = weighted_interpolate(fused * masks["R"], masks["R"], radius=2)
    g = weighted_interpolate(fused * masks["G"], masks["G"], radius=1)
    b = weighted_interpolate(fused * masks["B"], masks["B"], radius=2)
    rgb = np.stack([r, g, b], axis=2)
    measured = masks["R"] + masks["G"] + masks["B"]
    support = box_filter(measured, radius=1)
    demosaic_conf = np.clip(0.5 + support / max(float(np.max(support)), EPS) * 0.5, 0.0, 1.0)
    return {
        "rgb": rgb,
        "luma": g,
        "r_minus_g": r - g,
        "b_minus_g": b - g,
        "demosaic_confidence": demosaic_conf,
        "demosaic_method": method,
        "clear_channel": np.zeros_like(fused),
        "ir_channel": np.zeros_like(fused),
    }


def _decode_bayer_edge_aware(fused: ArrayF, masks: Mapping[str, ArrayF], *, method: str) -> Dict[str, Any]:
    values = np.asarray(fused, dtype=np.float64)
    red_mask = np.asarray(masks["R"], dtype=np.float64)
    green_mask = np.asarray(masks["G"], dtype=np.float64)
    blue_mask = np.asarray(masks["B"], dtype=np.float64)
    left = _shift2d(values, 0, -1)
    right = _shift2d(values, 0, 1)
    up = _shift2d(values, -1, 0)
    down = _shift2d(values, 1, 0)
    horizontal = 0.5 * (left + right)
    vertical = 0.5 * (up + down)
    grad_h = np.abs(left - right)
    grad_v = np.abs(up - down)
    green_interp = np.where(
        grad_h < grad_v,
        horizontal,
        np.where(grad_v < grad_h, vertical, 0.5 * (horizontal + vertical)),
    )
    g = np.where(green_mask > 0.0, values, green_interp)

    r_delta = weighted_interpolate((values - g) * red_mask, red_mask, radius=2)
    b_delta = weighted_interpolate((values - g) * blue_mask, blue_mask, radius=2)
    r = np.where(red_mask > 0.0, values, g + r_delta)
    b = np.where(blue_mask > 0.0, values, g + b_delta)

    measured = red_mask + green_mask + blue_mask
    support = box_filter(measured, radius=1)
    direction_gap = np.abs(grad_h - grad_v) / (grad_h + grad_v + EPS)
    local_contrast = np.maximum(grad_h, grad_v)
    contrast_norm = local_contrast / max(float(np.percentile(local_contrast, 98.0)), EPS)
    demosaic_conf = np.clip(
        0.42
        + 0.28 * support / max(float(np.max(support)), EPS)
        + 0.22 * direction_gap
        + 0.08 * (1.0 - np.clip(contrast_norm, 0.0, 1.0)),
        0.0,
        1.0,
    )
    rgb = np.stack([r, g, b], axis=2)
    return {
        "rgb": rgb,
        "luma": g,
        "r_minus_g": r - g,
        "b_minus_g": b - g,
        "demosaic_confidence": demosaic_conf,
        "demosaic_method": method,
        "clear_channel": np.zeros_like(values),
        "ir_channel": np.zeros_like(values),
    }


def _shift2d(values: ArrayF, row_delta: int, col_delta: int) -> ArrayF:
    array = np.asarray(values, dtype=np.float64)
    padded = np.pad(array, ((1, 1), (1, 1)), mode="edge")
    row0 = 1 + int(row_delta)
    col0 = 1 + int(col_delta)
    return padded[row0 : row0 + array.shape[0], col0 : col0 + array.shape[1]]


def _suppress_demosaic_artifacts(rgb: ArrayF, edge_confidence: ArrayF, demosaic_confidence: ArrayF, strength: float) -> ArrayF:
    values = np.asarray(rgb, dtype=np.float64)
    amount = np.clip(float(strength), 0.0, 1.0)
    if amount <= EPS:
        return values.copy()
    smooth = box_filter(values, radius=1)
    high_frequency = np.mean(np.abs(values[:, :, :3] - smooth[:, :, :3]), axis=2)
    hf_norm = high_frequency / max(float(np.percentile(high_frequency, 98.0)), EPS)
    artifact_likelihood = np.clip(0.35 + 0.65 * hf_norm + 0.25 * (1.0 - demosaic_confidence), 0.0, 1.0)
    edge_gate = np.clip(1.0 - 0.35 * edge_confidence, 0.35, 1.0)
    blend = np.clip(amount * artifact_likelihood * edge_gate, 0.0, 0.85)
    return values * (1.0 - blend[:, :, None]) + smooth * blend[:, :, None]


def _tone_map_rgb(rgb: ArrayF, mode: str, gamma: float) -> ArrayF:
    normalized = str(mode or "log").lower().replace("-", "_")
    values = np.asarray(rgb, dtype=np.float64)
    if normalized in {"log", "perception_log"}:
        return log_tonemap(values)
    if normalized in {"human_log", "display_log", "detector_log", "detector_safe_log"}:
        return gamma_encode(log_tonemap(values), gamma)
    if normalized in {"srgb", "detector_srgb", "display_srgb"}:
        return gamma_encode(_auto_level_rgb(values), gamma)
    if normalized in {"gamma", "gamma_srgb"}:
        return gamma_encode(np.clip(values, 0.0, 1.0), gamma)
    if normalized in {"linear", "none"}:
        return np.clip(values, 0.0, 1.0)
    return log_tonemap(values)


def _auto_level_rgb(rgb: ArrayF) -> ArrayF:
    values = np.asarray(rgb, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low = float(np.percentile(finite, 0.5))
    high = float(np.percentile(finite, 99.5))
    if high <= low:
        return np.clip(values, 0.0, 1.0)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _decode_clear_cfa(fused: ArrayF, masks: Mapping[str, ArrayF], pattern: str) -> Dict[str, Any]:
    r = weighted_interpolate(fused * masks["R"], masks["R"], radius=2)
    clear_mask = masks["C"]
    c = weighted_interpolate(fused * clear_mask, clear_mask, radius=1)
    if np.any(masks["B"]):
        b = weighted_interpolate(fused * masks["B"], masks["B"], radius=2)
    else:
        b = c * 0.92
    if np.any(masks["G"]):
        g = weighted_interpolate(fused * masks["G"], masks["G"], radius=2)
    else:
        g = c
    rgb = np.stack([r, g, b], axis=2)
    color_uncertainty = np.clip(1.0 - (masks["R"] + masks["B"] + masks["G"]), 0.0, 1.0)
    demosaic_conf = np.clip(1.0 - 0.4 * color_uncertainty, 0.0, 1.0)
    return {
        "rgb": rgb,
        "luma": c,
        "r_minus_g": r - g,
        "b_minus_g": b - g,
        "demosaic_confidence": demosaic_conf,
        "clear_channel": c,
        "color_uncertainty": color_uncertainty,
        "low_light_confidence": c / (c + 0.05),
        "ir_channel": np.zeros_like(fused),
    }


def _decode_rgb_ir(
    fused: ArrayF,
    masks: Mapping[str, ArrayF],
    calibration: CalibrationProfile,
) -> Dict[str, Any]:
    r = weighted_interpolate(fused * masks["R"], masks["R"], radius=2)
    g = weighted_interpolate(fused * masks["G"], masks["G"], radius=2)
    b = weighted_interpolate(fused * masks["B"], masks["B"], radius=2)
    ir = weighted_interpolate(fused * masks["IR"], masks["IR"], radius=2)
    stack = np.stack([r, g, b, ir], axis=2)
    corrected = np.tensordot(stack, calibration.rgb_ir_crosstalk.T, axes=([-1], [0]))
    corrected = np.clip(corrected, 0.0, 4.0)
    luma = np.mean(corrected, axis=2)
    contamination = np.clip(ir / (luma + ir + EPS), 0.0, 1.0)
    return {
        "rgb": corrected,
        "luma": luma,
        "r_minus_g": corrected[:, :, 0] - corrected[:, :, 1],
        "b_minus_g": corrected[:, :, 2] - corrected[:, :, 1],
        "demosaic_confidence": np.clip(1.0 - 0.25 * contamination, 0.0, 1.0),
        "ir_channel": ir,
        "ir_confidence": np.clip(ir / (ir + 0.05), 0.0, 1.0),
        "ir_contamination_map": contamination,
        "clear_channel": np.zeros_like(fused),
    }


def _decode_monochrome(fused: ArrayF) -> Dict[str, Any]:
    rgb = np.repeat(fused[:, :, None], 3, axis=2)
    return {
        "rgb": rgb,
        "luma": fused,
        "r_minus_g": np.zeros_like(fused),
        "b_minus_g": np.zeros_like(fused),
        "demosaic_confidence": np.ones_like(fused),
        "clear_channel": fused,
        "ir_channel": np.zeros_like(fused),
    }


def _blur_focus_confidence(edge_strength: ArrayF, edge_confidence: ArrayF, edge_percentile: float) -> ArrayF:
    threshold = percentile(edge_strength, edge_percentile)
    if threshold <= EPS:
        return np.zeros_like(edge_strength)
    local = edge_strength / threshold
    return np.clip(0.5 * local + 0.5 * edge_confidence, 0.0, 1.0)


def _combined_edge_evidence(edge_confidence: ArrayF, edge_strength: ArrayF) -> ArrayF:
    confidence = _robust_unit_map(edge_confidence)
    strength = _robust_unit_map(edge_strength)
    return np.sqrt(np.clip(confidence * strength, 0.0, 1.0))


def _robust_unit_map(values: ArrayF) -> ArrayF:
    arr = np.asarray(values, dtype=np.float64)
    low = float(np.percentile(arr, 1.0))
    high = float(np.percentile(arr, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.clip(arr, 0.0, 1.0)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _psf_blur_confidence(psf_sigma: ArrayF) -> ArrayF:
    sigma = np.maximum(np.asarray(psf_sigma, dtype=np.float64), 0.0)
    return np.clip(np.exp(-0.5 * sigma * sigma), 0.0, 1.0)


def _fast_roi(shape: Tuple[int, int], mode: str, fraction: float) -> Tuple[int, int, int, int]:
    rows, cols = int(shape[0]), int(shape[1])
    frac = np.clip(float(fraction), 0.05, 1.0)
    height = max(1, int(round(rows * frac)))
    mode_norm = str(mode).lower().replace("-", "_")
    if mode_norm == "top":
        return (0, 0, height, cols)
    if mode_norm == "center":
        start = max((rows - height) // 2, 0)
        return (start, 0, min(start + height, rows), cols)
    return (max(rows - height, 0), 0, rows, cols)


def _attach_internal_maps(result: PerceptionISPResult) -> PerceptionISPResult:
    return result
