"""Executable, visual examples for the PerceptionISP front-end contracts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import html as html_lib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from perception_isp.core.aux_dnn import (
    RGB_AUX_CHANNELS,
    RGB_AUX_EXTENDED_CHANNELS,
    build_rgb_aux_extended_tensor,
    build_rgb_aux_tensor,
)
from perception_isp.core.camerae2e_bridge import raw_from_camerae2e
from perception_isp.core.paths import output_root
from perception_isp.core.pipeline import PerceptionISPPipeline
from perception_isp.core.synthetic import make_synthetic_raw, make_synthetic_scene_rgb
from perception_isp.core.task_types import EvaluationSample
from perception_isp.core.types import (
    CalibrationProfile,
    PerceptionISPConfig,
    PreviousFrameState,
    RawFrame,
    SensorMetadata,
    json_ready,
)
from perception_isp.evaluation.comparison import build_pipeline_images
from perception_isp.evaluation.edge_confidence_suite import build_edge_confidence_suite
from perception_isp.evaluation.mechanism_validation import build_mechanism_validation


SUMMARY_NAME = "summary.json"
CASE_GROUPS = ("hdr", "metadata", "calibration", "cfa-optics", "temporal", "dnn-contract")
SECTION_ORDER = CASE_GROUPS + ("camerae2e",)
SECTION_TITLES = {
    "hdr": "HDR",
    "metadata": "Sensor Metadata",
    "calibration": "Calibration",
    "cfa-optics": "CFA / Geometry",
    "temporal": "Temporal",
    "dnn-contract": "DNN Contract",
    "camerae2e": "CameraE2E",
}


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run the self-contained PerceptionISP example suite.")
    parser.add_argument("--case", action="append", choices=CASE_GROUPS, default=None, help="Run only this case group. Repeatable.")
    parser.add_argument("--list-cases", action="store_true", help="List available case groups and exit.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--with-camerae2e", action="store_true", help="Require and validate the CameraE2E integration.")
    parser.add_argument("--scene", default="uniform ee", help="CameraE2E scene used with --with-camerae2e.")
    parser.add_argument("--output-dir", default=str(output_root() / "perception_isp_example_suite"))
    args = parser.parse_args(argv)

    if args.list_cases:
        for name in CASE_GROUPS:
            print(name)
        return 0

    summary = build_example_suite(
        width=int(args.width),
        height=int(args.height),
        seed=int(args.seed),
        case_groups=tuple(args.case or CASE_GROUPS),
        with_camerae2e=bool(args.with_camerae2e),
        scene_name=str(args.scene),
    )
    report = write_example_suite(summary, args.output_dir)
    payload = {
        "status": summary["status"],
        "report": str(report),
        "summary_json": str(report.parent / SUMMARY_NAME),
        "failed_sections": [row["id"] for row in summary["sections"] if row["status"] in {"fail", "error"}],
    }
    print(json.dumps(json_ready(payload), indent=2))
    return 0 if summary["status"] == "pass" else 1


def build_example_suite(
    *,
    width: int = 320,
    height: int = 180,
    seed: int = 7,
    case_groups: Sequence[str] = CASE_GROUPS,
    with_camerae2e: bool = False,
    scene_name: str = "uniform ee",
) -> Dict[str, Any]:
    width = max(int(width), 32)
    height = max(int(height), 24)
    selected = tuple(dict.fromkeys(str(value) for value in case_groups))
    unknown = sorted(set(selected) - set(CASE_GROUPS))
    if unknown:
        raise ValueError(f"unsupported example case groups: {', '.join(unknown)}")

    builders: Mapping[str, Callable[..., Dict[str, Any]]] = {
        "hdr": _build_hdr_section,
        "metadata": _build_metadata_section,
        "calibration": _build_calibration_section,
        "cfa-optics": _build_cfa_optics_section,
        "temporal": _build_temporal_section,
        "dnn-contract": _build_dnn_contract_section,
    }
    sections = []
    for section_id in CASE_GROUPS:
        if section_id not in selected:
            sections.append(_not_run_section(section_id))
            continue
        try:
            sections.append(builders[section_id](width=width, height=height, seed=seed))
        except Exception as exc:
            sections.append(_error_section(section_id, exc))
    sections.append(
        _build_camerae2e_section(
            width=width,
            height=height,
            scene_name=scene_name,
            required=bool(with_camerae2e),
        )
    )

    required = [row for row in sections if row["id"] in selected or (row["id"] == "camerae2e" and with_camerae2e)]
    status = "pass" if required and all(row["status"] == "pass" for row in required) else "fail"
    return {
        "title": "PerceptionISP Executable Example Suite",
        "status": status,
        "width": width,
        "height": height,
        "seed": int(seed),
        "selected_case_groups": list(selected),
        "camerae2e_required": bool(with_camerae2e),
        "sections": sections,
        "interpretation": (
            "This suite validates controlled front-end mechanisms and DNN input contracts. "
            "It is not evidence that PerceptionISP outperforms a HumanISP on a held-out perception task."
        ),
        "known_limitations": [
            "Scalar SensorMetadata is exported as sidecar metadata and is not part of the RGB+Aux DNN tensor.",
            "The saturation and clipping-distance maps describe exposure risk, not successful final HDR recovery.",
            "CameraE2E integration currently derives a synthetic exposure bracket from one simulated sensor output.",
            "Rolling-shutter row timing is modeled, but geometric motion compensation is not implemented.",
        ],
    }


def write_example_suite(summary: Mapping[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    serializable = _materialize_assets(summary, destination)
    (destination / SUMMARY_NAME).write_text(json.dumps(json_ready(serializable), indent=2) + "\n")
    html_path = destination / "index.html"
    html_path.write_text(_render_html(serializable))
    return html_path


def _build_hdr_section(*, width: int, height: int, seed: int) -> Dict[str, Any]:
    truth, static_planes, motion_planes, scales = _controlled_hdr_planes(width, height, seed)
    long_raw = _raw_from_encoded_planes(static_planes[:1], scales[:1], width, height, "hdr_single_long")
    short_raw = _raw_from_encoded_planes(static_planes[-1:], scales[-1:], width, height, "hdr_single_short")
    bracket_raw = _raw_from_encoded_planes(static_planes, scales, width, height, "hdr_static_bracket")
    motion_raw = _raw_from_encoded_planes(motion_planes, scales, width, height, "hdr_motion_bracket")
    pipeline = PerceptionISPPipeline()
    long_result = pipeline.run(long_raw)
    short_result = pipeline.run(short_raw)
    bracket_result = pipeline.run(bracket_raw)
    motion_result = pipeline.run(motion_raw)

    highlight = truth >= 0.90
    dark = (truth >= 0.015) & (truth <= 0.15)
    long_estimate = long_result.raw_normalized / scales[0]
    short_estimate = short_result.raw_normalized / scales[-1]
    bracket_estimate = bracket_result.raw_normalized
    long_highlight_mae = _masked_mae(long_estimate, truth, highlight)
    short_dark_mae = _masked_mae(short_estimate, truth, dark)
    bracket_highlight_mae = _masked_mae(bracket_estimate, truth, highlight)
    bracket_dark_mae = _masked_mae(bracket_estimate, truth, dark)
    source_count = int(np.unique(bracket_result.maps["hdr_exposure_source"]).size)
    static_ghost = _mean(bracket_result.maps["ghost_motion_artifact"])
    motion_ghost = _mean(motion_result.maps["ghost_motion_artifact"])

    checks = [
        _check(
            "bracket_recovers_highlights",
            bracket_highlight_mae < long_highlight_mae,
            "Static bracket should reduce known-radiance highlight error relative to a long exposure.",
            f"bracket={bracket_highlight_mae:.6f}, long={long_highlight_mae:.6f}",
        ),
        _check(
            "bracket_improves_dark_region",
            bracket_dark_mae < short_dark_mae,
            "Static bracket should reduce dark-region error relative to a short exposure.",
            f"bracket={bracket_dark_mae:.6f}, short={short_dark_mae:.6f}",
        ),
        _check(
            "multiple_exposure_sources_selected",
            source_count >= 2,
            "HDR fusion should select more than one exposure across the scene.",
            f"unique_sources={source_count}",
        ),
        _check(
            "motion_raises_ghost_risk",
            motion_ghost > static_ghost,
            "A shifted bracket should raise the ghost-motion artifact map.",
            f"motion={motion_ghost:.6f}, static={static_ghost:.6f}",
        ),
    ]
    cases = [
        _result_case(
            "single_long",
            "Single long exposure: good shadow signal with clipped highlights.",
            long_result,
            metrics={"highlight_radiance_mae": long_highlight_mae, "exposure_scale": scales[0]},
            extra_assets={
                "input_raw": static_planes[0],
                "hdr_exposure_source": long_result.maps["hdr_exposure_source"],
                "saturation_risk": long_result.maps["saturation"],
            },
        ),
        _result_case(
            "single_short",
            "Single short exposure: highlight protection with amplified dark-region noise.",
            short_result,
            metrics={"dark_radiance_mae": short_dark_mae, "exposure_scale": scales[-1]},
            extra_assets={
                "input_raw": static_planes[-1],
                "hdr_exposure_source": short_result.maps["hdr_exposure_source"],
                "saturation_risk": short_result.maps["saturation"],
            },
        ),
        _result_case(
            "static_bracket",
            "Static three-exposure HDR bracket.",
            bracket_result,
            metrics={
                "highlight_radiance_mae": bracket_highlight_mae,
                "dark_radiance_mae": bracket_dark_mae,
                "unique_exposure_sources": source_count,
                "ghost_motion_mean": static_ghost,
            },
            extra_assets={
                "known_radiance": truth,
                "fused_radiance": bracket_estimate,
                "hdr_exposure_source": bracket_result.maps["hdr_exposure_source"],
                "hdr_confidence": bracket_result.maps["hdr_confidence"],
                "saturation_risk": bracket_result.maps["saturation"],
                "ghost_motion_artifact": bracket_result.maps["ghost_motion_artifact"],
            },
        ),
        _result_case(
            "motion_bracket",
            "Three-exposure bracket with controlled inter-exposure motion.",
            motion_result,
            metrics={"ghost_motion_mean": motion_ghost},
            extra_assets={
                "hdr_exposure_source": motion_result.maps["hdr_exposure_source"],
                "ghost_motion_artifact": motion_result.maps["ghost_motion_artifact"],
            },
        ),
    ]
    return _section(
        "hdr",
        checks,
        cases,
        "Known scene radiance separates actual HDR recovery from exposure-risk maps.",
        notes=[
            "saturation means any source exposure crossed the saturation threshold.",
            "clipping_distance is measured from the brightest source exposure and is not a fused-output clipping metric.",
        ],
    )


def _build_metadata_section(*, width: int, height: int, seed: int) -> Dict[str, Any]:
    base_raw = make_synthetic_raw(width=width, height=height, exposures=(0.08,), seed=seed + 101)
    base_raw.metadata = replace(base_raw.metadata, frame_counter=41, timestamp_us=1_000_000.0, temperature_c=25.0)
    pipeline = PerceptionISPPipeline()
    base = pipeline.run(base_raw)

    black = float(base_raw.calibration.black_level)
    white = float(base_raw.calibration.white_level)
    base_plane = np.asarray(base_raw.data, dtype=np.float64)[0]
    gain_two_plane = np.clip((base_plane - black) * 2.0 + black, black, white)
    gain_raw = replace(
        base_raw,
        data=np.stack([base_plane, gain_two_plane], axis=0),
        metadata=replace(
            base_raw.metadata,
            exposure_times_us=(640.0, 640.0),
            analog_gains=(1.0, 2.0),
            digital_gains=(1.0, 1.0),
            hdr_mode="multi_exposure",
            hdr_ratios=(1.0, 1.0),
        ),
    )
    gain_result = pipeline.run(gain_raw)
    gain_error = float(np.max(np.abs(gain_result.raw_normalized - base.raw_normalized)))

    hot_raw = replace(base_raw, metadata=replace(base_raw.metadata, temperature_c=80.0))
    hot = pipeline.run(hot_raw)
    fast_timing_raw = replace(base_raw, metadata=replace(base_raw.metadata, line_time_us=10.0))
    slow_timing_raw = replace(base_raw, metadata=replace(base_raw.metadata, line_time_us=40.0))
    fast_timing = pipeline.run(fast_timing_raw)
    slow_timing = pipeline.run(slow_timing_raw)

    mismatch_rejected = _raises_value_error(
        replace(base_raw, calibration=replace(base_raw.calibration, cfa_pattern="GRBG")),
        "sensor metadata CFA does not match calibration CFA",
    )
    length_rejected = _raises_value_error(
        replace(gain_raw, metadata=replace(gain_raw.metadata, analog_gains=(1.0, 2.0, 3.0))),
        "analog_gains must contain one value or one value per exposure",
    )
    nonpositive_gain_rejected = _raises_value_error(
        replace(base_raw, metadata=replace(base_raw.metadata, digital_gains=(0.0,))),
        "digital_gains values must be finite and positive",
    )
    nonpositive_exposure_rejected = _raises_value_error(
        replace(base_raw, metadata=replace(base_raw.metadata, exposure_times_us=(0.0,))),
        "exposure_times_us values must be finite and positive",
    )
    nonfinite_gain_rejected = _raises_value_error(
        replace(base_raw, metadata=replace(base_raw.metadata, analog_gains=(float("nan"),))),
        "analog_gains values must be finite and positive",
    )
    base_noise = _mean(base.maps["noise_variance"])
    hot_noise = _mean(hot.maps["noise_variance"])
    base_snr = _mean(base.maps["snr_map"])
    hot_snr = _mean(hot.maps["snr_map"])
    fast_span = _row_span(fast_timing)
    slow_span = _row_span(slow_timing)

    checks = [
        _check("gain_normalization_invariant", gain_error <= 1.0e-10, "Equivalent analog gain must not be applied twice.", f"max_abs_error={gain_error:.3e}"),
        _check("temperature_raises_noise", hot_noise > base_noise, "Hot-sensor metadata should raise modeled dark-current noise.", f"hot={hot_noise:.6f}, nominal={base_noise:.6f}"),
        _check("temperature_lowers_snr", hot_snr < base_snr, "Hot-sensor noise should lower the SNR confidence map.", f"hot={hot_snr:.6f}, nominal={base_snr:.6f}"),
        _check("line_time_changes_row_span", slow_span > fast_span, "Longer line time should increase the row timestamp span.", f"slow={slow_span:.1f}us, fast={fast_span:.1f}us"),
        _check("line_time_changes_fast_latency", slow_timing.fast.estimated_latency_us > fast_timing.fast.estimated_latency_us, "Longer line time should increase streaming fast-path latency.", f"slow={slow_timing.fast.estimated_latency_us:.1f}us, fast={fast_timing.fast.estimated_latency_us:.1f}us"),
        _check("cfa_mismatch_rejected", mismatch_rejected, "Sensor and calibration CFA mismatch must fail early.", "expected ValueError"),
        _check("metadata_length_rejected", length_rejected, "Per-exposure metadata must be scalar-broadcast or exact length.", "expected ValueError"),
        _check("nonpositive_gain_rejected", nonpositive_gain_rejected, "Gain metadata must be positive.", "expected ValueError"),
        _check("nonpositive_exposure_rejected", nonpositive_exposure_rejected, "Exposure-time metadata must be positive.", "expected ValueError"),
        _check("nonfinite_gain_rejected", nonfinite_gain_rejected, "Gain metadata must be finite.", "expected ValueError"),
        _check(
            "identity_metadata_roundtrip",
            int(base.metadata["frame"]["frame_counter"]) == 41 and float(base.metadata["frame"]["timestamp_us"]) == 1_000_000.0,
            "Frame identity and timing metadata must survive the pipeline.",
            f"frame={base.metadata['frame']['frame_counter']}, timestamp={base.metadata['frame']['timestamp_us']}",
        ),
    ]
    cases = [
        _result_case("metadata_nominal", "25 C nominal metadata baseline.", base, metrics={"noise_mean": base_noise, "snr_mean": base_snr}),
        _result_case("metadata_hot", "80 C metadata stress case.", hot, metrics={"noise_mean": hot_noise, "snr_mean": hot_snr}),
        _result_case("gain_equivalent_stack", "Matched 1x/2x gain planes after physical normalization.", gain_result, metrics={"max_radiance_error": gain_error}),
        _result_case(
            "slow_line_time",
            "40 us line-time timing case.",
            slow_timing,
            metrics={"row_span_us": slow_span, "fast_latency_us": slow_timing.fast.estimated_latency_us},
            extra_assets={"rolling_row_fraction": slow_timing.maps["rolling_row_fraction"]},
        ),
    ]
    return _section(
        "metadata",
        checks,
        cases,
        "Metadata is separated into active processing inputs, transport-only fields, and declared but unused calibration fields.",
        support_matrix=_metadata_support_matrix(),
    )


def _build_calibration_section(*, width: int, height: int, seed: int) -> Dict[str, Any]:
    mechanism = build_mechanism_validation(
        width=width,
        height=height,
        seed=seed + 201,
        cfa_patterns=("RGGB", "GRBG", "RCCB", "RGBIR"),
    )
    edge_validation = build_edge_confidence_suite(width=width, height=height, seed=seed + 202, cfa_pattern="RGGB")
    checks = [
        _check(
            f"mechanism_{row['id']}",
            row["status"] == "pass",
            str(row["description"]),
            _criteria_evidence(row.get("criteria", ())),
        )
        for row in mechanism["mechanisms"]
    ]
    checks.extend(
        _check(
            f"edge_{row['id']}",
            row["status"] == "pass",
            str(row["description"]),
            _criteria_evidence(row.get("criteria", ())),
        )
        for row in edge_validation["checks"]
    )

    raw = make_synthetic_raw(width=width, height=height, exposures=(0.25,), seed=seed + 203)
    row, col = height // 2, width // 2
    corrupted_data = np.asarray(raw.data, dtype=np.float64).copy()
    corrupted_data[:, row, col] = float(raw.calibration.white_level)
    uncorrected_raw = replace(raw, data=corrupted_data, calibration=replace(raw.calibration, defect_pixels=()))
    corrected_raw = replace(raw, data=corrupted_data, calibration=replace(raw.calibration, defect_pixels=((row, col),)))
    pipeline = PerceptionISPPipeline()
    uncorrected = pipeline.run(uncorrected_raw)
    corrected = pipeline.run(corrected_raw)
    uncorrected_residual = _center_residual(uncorrected.raw_normalized, row, col)
    corrected_residual = _center_residual(corrected.raw_normalized, row, col)

    psf_raw = make_synthetic_raw(width=width, height=height, seed=seed + 204)
    shape = (height, width)
    sharp = pipeline.run(replace(psf_raw, calibration=replace(psf_raw.calibration, psf_sigma_map=np.zeros(shape))))
    blurred = pipeline.run(
        replace(
            psf_raw,
            calibration=replace(
                psf_raw.calibration,
                psf_sigma_map=np.full(shape, 1.6),
                mtf_confidence_map=np.full(shape, 0.25),
            ),
        )
    )
    sharp_edge = _mean(sharp.maps["edge_confidence"])
    blurred_edge = _mean(blurred.maps["edge_confidence"])
    lens = np.asarray(sharp.maps["lens_gain"], dtype=np.float64)
    corner_gain = float(np.mean([lens[0, 0], lens[0, -1], lens[-1, 0], lens[-1, -1]]))
    center_gain = float(lens[height // 2, width // 2])

    black_raw = replace(
        raw,
        data=np.full((height, width), float(raw.calibration.black_level), dtype=np.float64),
        metadata=replace(raw.metadata, exposure_times_us=(8000.0,), analog_gains=(1.0,), digital_gains=(1.0,), hdr_ratios=(1.0,), hdr_mode="single"),
    )
    black_result = pipeline.run(black_raw)
    black_max = float(np.max(np.abs(black_result.raw_normalized)))
    checks.extend(
        [
            _check("black_level_maps_to_zero", black_max <= 1.0e-12, "Black-level code should normalize to zero.", f"max_abs={black_max:.3e}"),
            _check("defect_correction_reduces_outlier", corrected_residual < uncorrected_residual, "Defect correction should reduce the calibrated pixel outlier.", f"corrected={corrected_residual:.6f}, uncorrected={uncorrected_residual:.6f}"),
            _check("lens_shading_gain_is_spatial", corner_gain > center_gain, "Synthetic lens-shading gain should be larger at the image corners.", f"corner={corner_gain:.4f}, center={center_gain:.4f}"),
            _check("psf_mtf_reduce_edge_confidence", blurred_edge < sharp_edge, "Resolved PSF/MTF stress should reduce edge confidence.", f"blurred={blurred_edge:.6f}, sharp={sharp_edge:.6f}"),
        ]
    )
    cases = [row for row in mechanism["cases"] if row.get("group") != "cfa_support"]
    cases.extend(
        [
            _result_case("defect_uncorrected", "Injected hot pixel without a defect table.", uncorrected, metrics={"center_residual": uncorrected_residual}),
            _result_case("defect_corrected", "Injected hot pixel corrected by calibration metadata.", corrected, metrics={"center_residual": corrected_residual}),
            _result_case("psf_mtf_stress", "PSF sigma 1.6 px and MTF confidence 0.25.", blurred, metrics={"edge_confidence_mean": blurred_edge}),
        ]
    )
    return _section(
        "calibration",
        checks,
        cases,
        "Existing mechanism and edge-confidence validators are reused and extended with defect, black-level, lens, and PSF checks.",
        notes=["color_shading_gain is declared in the calibration contract but is not currently consumed by the pipeline."],
    )


def _build_cfa_optics_section(*, width: int, height: int, seed: int) -> Dict[str, Any]:
    patterns = ("RGGB", "GRBG", "BGGR", "GBRG", "RCCB", "RGBIR", "MONO")
    pipeline = PerceptionISPPipeline()
    cases = []
    pattern_checks = []
    for index, pattern in enumerate(patterns):
        raw = make_synthetic_raw(width=width, height=height, cfa_pattern=pattern, seed=seed + 300 + index)
        result = pipeline.run(raw)
        finite = bool(np.isfinite(result.vision_rgb).all() and all(np.isfinite(value).all() for value in result.maps.values()))
        preserved = str(result.metadata["frame"]["cfa_pattern"]) == pattern
        pattern_checks.append(
            _check(
                f"cfa_{pattern.lower()}",
                finite and preserved,
                f"{pattern} must preserve its source pattern and produce finite outputs.",
                f"finite={finite}, reported={result.metadata['frame']['cfa_pattern']}",
            )
        )
        cases.append(
            _result_case(
                f"cfa_{pattern.lower()}",
                f"{pattern} decode path.",
                result,
                metrics={"edge_confidence_mean": _mean(result.maps["edge_confidence"]), "color_confidence_mean": _mean(result.maps["color_confidence"])},
            )
        )

    geometry_raw = make_synthetic_raw(width=width, height=height, cfa_pattern="RGGB", seed=seed + 399)
    identity = PerceptionISPPipeline(PerceptionISPConfig(accurate_enable_dewarp=False)).run(geometry_raw)
    dewarped = PerceptionISPPipeline(PerceptionISPConfig(accurate_enable_dewarp=True)).run(geometry_raw)
    identity_rgb = identity.accurate.tensor[:, :, :3]
    dewarped_rgb = dewarped.accurate.tensor[:, :, :3]
    dewarp_delta = float(np.mean(np.abs(identity_rgb - dewarped_rgb)))
    pattern_checks.append(
        _check(
            "dewarp_changes_accurate_geometry",
            dewarp_delta > 1.0e-5 and dewarped.metadata["geometry"]["roi_coordinate_transform"] != "identity",
            "Enabling dewarp should change the accurate-path geometry and record its transform.",
            f"mean_abs_delta={dewarp_delta:.6f}, transform={dewarped.metadata['geometry']['roi_coordinate_transform']}",
        )
    )
    cases.append(
        _result_case(
            "accurate_dewarp",
            "Accurate path with radial distortion correction enabled.",
            dewarped,
            metrics={"accurate_rgb_mean_abs_delta": dewarp_delta},
            extra_assets={"accurate_rgb": dewarped_rgb, "identity_accurate_rgb": identity_rgb},
        )
    )
    return _section(
        "cfa-optics",
        pattern_checks,
        cases,
        "Source CFA is authoritative; alternate color-filter layouts and accurate-path dewarp are explicitly auditable.",
    )


def _build_temporal_section(*, width: int, height: int, seed: int) -> Dict[str, Any]:
    pipeline = PerceptionISPPipeline()
    first_raw = make_synthetic_raw(width=width, height=height, frame_counter=0, timestamp_us=0.0, seed=seed + 401)
    first = pipeline.run(first_raw)
    previous = PreviousFrameState(luma=first.next_state.luma, rgb=first.next_state.rgb, timestamp_us=0.0, frame_counter=0)

    second_raw = make_synthetic_raw(width=width, height=height, frame_counter=1, timestamp_us=33333.0, seed=seed + 401)
    second = pipeline.run(second_raw, previous)
    flash_data = np.asarray(second_raw.data, dtype=np.float64).copy()
    r0, r1 = max(height // 6, 0), max(height // 3, 1)
    c0, c1 = max((width * 3) // 4, 0), max((width * 7) // 8, 1)
    flash_data[:, r0:r1, c0:c1] = float(second_raw.calibration.white_level)
    flash_raw = replace(second_raw, data=flash_data, metadata=replace(second_raw.metadata, frame_counter=2, timestamp_us=66666.0))
    flash = pipeline.run(flash_raw, previous)

    bottom_raw = replace(first_raw, metadata=replace(first_raw.metadata, readout_direction="bottom_to_top"))
    bottom = pipeline.run(bottom_raw)
    temporal_mean = _mean(second.maps["temporal_difference"])
    base_flicker = _mean(first.maps["led_flicker_confidence"])
    flash_flicker = _mean(flash.maps["led_flicker_confidence"][r0:r1, c0:c1])
    top_fraction = np.asarray(first.maps["rolling_row_fraction"])
    bottom_fraction = np.asarray(bottom.maps["rolling_row_fraction"])
    bottom_geometry = bottom.metadata["geometry"]
    checks = [
        _check("previous_frame_generates_difference", temporal_mean > 0.0, "A changed second frame should produce temporal difference.", f"mean={temporal_mean:.6f}"),
        _check("flash_raises_led_flicker", flash_flicker > base_flicker, "A bright changed ROI should raise LED-flicker confidence.", f"flash_roi={flash_flicker:.6f}, first_frame={base_flicker:.6f}"),
        _check("top_readout_increases_by_row", float(top_fraction[0]) < float(top_fraction[-1]), "Top-to-bottom readout fraction should increase with row index.", f"first={top_fraction[0]:.1f}, last={top_fraction[-1]:.1f}"),
        _check("bottom_readout_reverses_by_row", float(bottom_fraction[0]) > float(bottom_fraction[-1]), "Bottom-to-top readout fraction should decrease with row index.", f"first={bottom_fraction[0]:.1f}, last={bottom_fraction[-1]:.1f}"),
        _check("bottom_row_timestamps_reverse", float(bottom_geometry["row0_timestamp_us"]) > float(bottom_geometry["row_last_timestamp_us"]), "Bottom-to-top readout should timestamp the last image row first.", f"row0={bottom_geometry['row0_timestamp_us']:.1f}, row_last={bottom_geometry['row_last_timestamp_us']:.1f}"),
        _check("next_state_preserves_identity", flash.next_state.frame_counter == 2 and flash.next_state.timestamp_us == 66666.0, "Temporal state must preserve frame identity and timestamp.", f"frame={flash.next_state.frame_counter}, timestamp={flash.next_state.timestamp_us}"),
    ]
    cases = [
        _result_case("temporal_first", "First frame without previous state.", first),
        _result_case(
            "temporal_second",
            "Second frame processed with previous state.",
            second,
            metrics={"temporal_difference_mean": temporal_mean},
            extra_assets={"temporal_difference": second.maps["temporal_difference"], "temporal_consistency": second.maps["temporal_consistency"]},
        ),
        _result_case(
            "temporal_flash",
            "Bright changed ROI used to exercise flicker confidence.",
            flash,
            metrics={"flash_roi_flicker_mean": flash_flicker},
            extra_assets={"led_flicker_confidence": flash.maps["led_flicker_confidence"], "temporal_difference": flash.maps["temporal_difference"]},
        ),
        _result_case(
            "bottom_to_top",
            "Bottom-to-top rolling readout metadata.",
            bottom,
            metrics={"row0_timestamp_us": bottom_geometry["row0_timestamp_us"], "row_last_timestamp_us": bottom_geometry["row_last_timestamp_us"]},
            extra_assets={"rolling_row_fraction": bottom.maps["rolling_row_fraction"]},
        ),
    ]
    return _section(
        "temporal",
        checks,
        cases,
        "Previous-frame state and row timing produce temporal confidence signals; geometric rolling-shutter compensation is not implemented.",
    )


def _build_dnn_contract_section(*, width: int, height: int, seed: int) -> Dict[str, Any]:
    raw = make_synthetic_raw(width=width, height=height, seed=seed + 501)
    sample = EvaluationSample(sample_id="dnn_contract", raw=raw, ground_truth=(), source="synthetic")
    images = build_pipeline_images(sample)
    stable = build_rgb_aux_tensor(images, layout="hwc")
    extended = build_rgb_aux_extended_tensor(images, layout="hwc")
    scalar_names = {
        "exposure_times_us",
        "analog_gains",
        "digital_gains",
        "temperature_c",
        "timestamp_us",
        "line_time_us",
        "hdr_mode",
        "hdr_ratios",
    }
    tensor_names = set(RGB_AUX_EXTENDED_CHANNELS)
    checks = [
        _check("stable_tensor_shape", stable.shape == (height, width, len(RGB_AUX_CHANNELS)), "Stable RGB+Aux tensor must expose the six-channel contract.", f"shape={list(stable.shape)}"),
        _check("extended_tensor_shape", extended.shape == (height, width, len(RGB_AUX_EXTENDED_CHANNELS)), "Extended tensor must expose all sixteen mapped channels.", f"shape={list(extended.shape)}"),
        _check("stable_tensor_finite_unit_range", _finite_unit_range(stable), "Stable DNN tensor must be finite and normalized to [0, 1].", f"min={float(np.min(stable)):.6f}, max={float(np.max(stable)):.6f}"),
        _check("extended_tensor_finite_unit_range", _finite_unit_range(extended), "Extended DNN tensor must be finite and normalized to [0, 1].", f"min={float(np.min(extended)):.6f}, max={float(np.max(extended)):.6f}"),
        _check("scalar_metadata_is_sidecar_only", scalar_names.isdisjoint(tensor_names), "Scalar SensorMetadata must remain sidecar-only in the current model contract.", "no scalar metadata channel is present"),
    ]
    cases = [
        {
            "id": "rgb_aux_contract",
            "description": "Current early-fusion tensor contract without a scalar metadata branch.",
            "status": "pass",
            "metadata": {"frame": dict(images.metadata.get("frame", {}))},
            "metrics": {
                "stable_shape": list(stable.shape),
                "stable_channels": list(RGB_AUX_CHANNELS),
                "extended_shape": list(extended.shape),
                "extended_channels": list(RGB_AUX_EXTENDED_CHANNELS),
                "scalar_metadata_in_tensor": False,
            },
            "_assets_source": {
                "perception_rgb": images.perception_rgb,
                "edge_strength": images.aux_maps["edge_strength"],
                "saturation_risk": images.aux_maps["saturation"],
                "snr_confidence": images.aux_maps["snr_map"],
                "hdr_confidence": images.aux_maps["hdr_confidence"],
                "psf_edge_likelihood": images.aux_maps["psf_edge_likelihood"],
            },
        }
    ]
    return _section(
        "dnn-contract",
        checks,
        cases,
        "RGB and spatial Aux maps enter the DNN tensor. Scalar metadata remains auditable sidecar information until a dedicated model branch is trained.",
    )


def _build_camerae2e_section(*, width: int, height: int, scene_name: str, required: bool) -> Dict[str, Any]:
    if not required:
        return {
            "id": "camerae2e",
            "title": SECTION_TITLES["camerae2e"],
            "status": "skip",
            "interpretation": "CameraE2E was not requested. Re-run with --with-camerae2e to require this integration.",
            "checks": [],
            "cases": [],
            "notes": ["No synthetic fallback is accepted as a CameraE2E success."],
        }
    try:
        raw = raw_from_camerae2e(scene_name=scene_name, width=width, height=height, cfa_pattern="auto")
        result = PerceptionISPPipeline().run(raw)
    except Exception as exc:
        return {
            "id": "camerae2e",
            "title": SECTION_TITLES["camerae2e"],
            "status": "fail",
            "interpretation": "CameraE2E was required but could not produce a directly auditable RAW frame.",
            "checks": [_check("camerae2e_direct_run", False, "Direct CameraE2E execution must succeed without fallback.", str(exc))],
            "cases": [],
            "notes": ["Install the camerae2e extra and set CAMERAE2E_ROOT before retrying."],
        }

    provenance = dict(raw.provenance)
    checks = [
        _check("camerae2e_direct_run", provenance.get("camerae2e_used") is True, "Direct CameraE2E execution must be recorded.", f"bridge={provenance.get('bridge')}"),
        _check("native_sensor_mosaic", provenance.get("true_sensor_cfa_mosaic") is True, "The bridge should consume a true 2-D sensor mosaic.", f"raw_source={provenance.get('raw_source_key')}"),
        _check("native_cfa_preserved", provenance.get("source_cfa_pattern") == provenance.get("target_cfa_pattern") and not provenance.get("pattern_remapped"), "Auto CFA should preserve the source sensor pattern.", f"source={provenance.get('source_cfa_pattern')}, target={provenance.get('target_cfa_pattern')}, remapped={provenance.get('pattern_remapped')}"),
        _check("native_resolution_audited", bool(provenance.get("native_resolution_at_least_target")), "Native sensor resolution should be at least the requested output resolution.", f"source={provenance.get('source_native_hw')}, target={provenance.get('target_shape')}"),
    ]
    cases = [_result_case("camerae2e_native", f"CameraE2E scene: {scene_name}", result, metrics={"provenance": provenance})]
    return _section(
        "camerae2e",
        checks,
        cases,
        "CameraE2E provenance is verified directly. The current bridge creates its HDR stack by scaling one simulated sensor mosaic.",
        notes=["This is a native-CFA bridge check, not true multi-capture CameraE2E HDR evidence."],
    )


def _section(
    section_id: str,
    checks: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    interpretation: str,
    *,
    notes: Sequence[str] = (),
    support_matrix: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    return {
        "id": section_id,
        "title": SECTION_TITLES[section_id],
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "interpretation": interpretation,
        "checks": list(checks),
        "cases": list(cases),
        "notes": list(notes),
        "support_matrix": list(support_matrix),
    }


def _check(identifier: str, passed: bool, description: str, evidence: str) -> Dict[str, Any]:
    return {
        "id": str(identifier),
        "status": "pass" if bool(passed) else "fail",
        "description": str(description),
        "evidence": str(evidence),
    }


def _result_case(
    identifier: str,
    description: str,
    result: Any,
    *,
    metrics: Mapping[str, Any] | None = None,
    extra_assets: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    case_metrics = {
        "snr_mean": _mean(result.maps["snr_map"]),
        "noise_variance_mean": _mean(result.maps["noise_variance"]),
        "hdr_confidence_mean": _mean(result.maps["hdr_confidence"]),
        "saturation_risk_mean": _mean(result.maps["saturation"]),
        "edge_confidence_mean": _mean(result.maps["edge_confidence"]),
        "visibility_confidence": float(result.health.get("visibility_confidence", 0.0)),
    }
    case_metrics.update(dict(metrics or {}))
    assets = {
        "human_rgb": result.human_rgb if result.human_rgb is not None else result.vision_rgb,
        "perception_rgb": result.vision_rgb,
        "noise_variance": result.maps["noise_variance"],
        "edge_confidence": result.maps["edge_confidence"],
    }
    assets.update(dict(extra_assets or {}))
    return {
        "id": str(identifier),
        "description": str(description),
        "status": "pass" if bool(np.isfinite(result.vision_rgb).all()) else "fail",
        "warnings": list(result.health.get("warnings", ())),
        "metadata": {
            "frame": dict(result.metadata.get("frame", {})),
            "calibration": dict(result.metadata.get("calibration", {})),
            "geometry": dict(result.metadata.get("geometry", {})),
            "raw_provenance": dict(result.metadata.get("raw_provenance", {})),
            "runtime_control_suggestion": dict(result.metadata.get("runtime_control_suggestion", {})),
        },
        "metrics": case_metrics,
        "_assets_source": assets,
    }


def _controlled_hdr_planes(width: int, height: int, seed: int) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[float, ...]]:
    rgb = make_synthetic_scene_rgb(width=width, height=height, seed=seed)
    truth = _mosaic_rggb(rgb)
    scales = (1.0, 0.25, 0.0625)
    static_rng = np.random.default_rng(seed + 11)
    motion_rng = np.random.default_rng(seed + 11)
    static_planes = tuple(_encode_sensor_plane(truth, scale, static_rng) for scale in scales)
    shifts = ((0, 0), (1, 3), (-2, -5))
    motion_planes = tuple(
        _encode_sensor_plane(_shift_no_wrap(truth, row_shift, col_shift), scale, motion_rng)
        for scale, (row_shift, col_shift) in zip(scales, shifts)
    )
    return truth, static_planes, motion_planes, scales


def _raw_from_encoded_planes(
    planes: Sequence[np.ndarray],
    scales: Sequence[float],
    width: int,
    height: int,
    label: str,
) -> RawFrame:
    scale_values = tuple(float(value) for value in scales)
    metadata = SensorMetadata(
        camera_id="example_suite",
        sensor_id="controlled_hdr_sensor",
        module_serial=label,
        calibration_id="controlled_hdr_v1",
        isp_profile_id="perception_isp_example",
        exposure_times_us=tuple(8000.0 * value for value in scale_values),
        analog_gains=tuple(1.0 for _ in scale_values),
        digital_gains=tuple(1.0 for _ in scale_values),
        hdr_mode="multi_exposure" if len(scale_values) > 1 else "single",
        hdr_ratios=scale_values,
        cfa_pattern="RGGB",
        rolling_shutter_time_us=33333.0,
        line_time_us=33333.0 / float(height),
    )
    calibration = CalibrationProfile(cfa_pattern="RGGB", black_level=64.0, white_level=4095.0)
    data = np.stack(tuple(np.asarray(value, dtype=np.float64) for value in planes), axis=0)
    return RawFrame(data=data, metadata=metadata, calibration=calibration, provenance={"source": "controlled_example", "case": label})


def _encode_sensor_plane(radiance: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.clip(np.asarray(radiance, dtype=np.float64) * float(scale), 0.0, 1.0)
    shot = rng.normal(0.0, np.sqrt(np.maximum(signal, 0.0)) * 0.006, size=signal.shape)
    read = rng.normal(0.0, 0.0015, size=signal.shape)
    noisy = np.clip(signal + shot + read, 0.0, 1.0)
    return np.round(noisy * (4095.0 - 64.0) + 64.0)


def _mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    source = np.asarray(rgb, dtype=np.float64)
    mosaic = np.zeros(source.shape[:2], dtype=np.float64)
    mosaic[0::2, 0::2] = source[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = source[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = source[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = source[1::2, 1::2, 2]
    return mosaic


def _shift_no_wrap(image: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    shifted = np.roll(source, (int(row_shift), int(col_shift)), axis=(0, 1))
    if row_shift > 0:
        shifted[:row_shift, :] = 0.0
    elif row_shift < 0:
        shifted[row_shift:, :] = 0.0
    if col_shift > 0:
        shifted[:, :col_shift] = 0.0
    elif col_shift < 0:
        shifted[:, col_shift:] = 0.0
    return shifted


def _metadata_support_matrix() -> list[Dict[str, str]]:
    rows = [
        ("exposure_times_us", "active processing", "HDR radiance normalization"),
        ("analog_gains / digital_gains", "active processing", "RAW physical normalization"),
        ("temperature_c", "active processing", "dark-current noise model"),
        ("cfa_pattern", "active processing", "CFA decoder and provenance contract"),
        ("timestamp_us", "active processing", "row timing and temporal frame delta"),
        ("line_time_us / readout_direction", "active processing", "row timing and fast latency"),
        ("frame_counter", "propagated only", "frame identity and previous-state traceability"),
        ("camera_id", "propagated only", "mapped to camera synchronization group metadata"),
        ("hdr_mode / hdr_ratios", "propagated only", "validated and recorded; fusion uses exposure planes and times"),
        ("rolling_shutter_time_us", "propagated only", "recorded in health metadata; compensation is not implemented"),
        ("sensor/module/profile IDs", "propagated only", "traceability and calibration selection sidecar"),
        ("color_shading_gain", "declared but unused", "calibration field exists but no pipeline block consumes it"),
    ]
    return [{"field": field, "status": status, "behavior": behavior} for field, status, behavior in rows]


def _raises_value_error(raw: RawFrame, message: str) -> bool:
    try:
        PerceptionISPPipeline().run(raw)
    except ValueError as exc:
        return message in str(exc)
    return False


def _criteria_evidence(rows: Sequence[Mapping[str, Any]]) -> str:
    passed = sum(1 for row in rows if row.get("pass") is True)
    return f"criteria_passed={passed}/{len(rows)}"


def _center_residual(image: np.ndarray, row: int, col: int) -> float:
    arr = np.asarray(image, dtype=np.float64)
    r0, r1 = max(int(row) - 1, 0), min(int(row) + 2, arr.shape[0])
    c0, c1 = max(int(col) - 1, 0), min(int(col) + 2, arr.shape[1])
    patch = arr[r0:r1, c0:c1].reshape(-1)
    center_index = (int(row) - r0) * (c1 - c0) + (int(col) - c0)
    neighbors = np.delete(patch, center_index)
    return float(abs(arr[int(row), int(col)] - np.median(neighbors)))


def _row_span(result: Any) -> float:
    geometry = result.metadata["geometry"]
    return float(geometry["row_timestamp_end_us"] - geometry["row_timestamp_start_us"])


def _masked_mae(value: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not bool(np.any(selected)):
        return 0.0
    return float(np.mean(np.abs(np.asarray(value)[selected] - np.asarray(truth)[selected])))


def _mean(value: Any) -> float:
    return float(np.mean(np.asarray(value, dtype=np.float64)))


def _finite_unit_range(value: np.ndarray) -> bool:
    arr = np.asarray(value, dtype=np.float64)
    return bool(np.isfinite(arr).all() and float(np.min(arr)) >= 0.0 and float(np.max(arr)) <= 1.0)


def _not_run_section(section_id: str) -> Dict[str, Any]:
    return {
        "id": section_id,
        "title": SECTION_TITLES[section_id],
        "status": "not_run",
        "interpretation": "This case group was not selected for the current run.",
        "checks": [],
        "cases": [],
        "notes": [],
        "support_matrix": [],
    }


def _error_section(section_id: str, exc: Exception) -> Dict[str, Any]:
    return {
        "id": section_id,
        "title": SECTION_TITLES[section_id],
        "status": "error",
        "interpretation": "The section raised an unexpected exception.",
        "checks": [_check("section_execution", False, "Section execution must complete.", f"{type(exc).__name__}: {exc}")],
        "cases": [],
        "notes": [],
        "support_matrix": [],
    }


def _materialize_assets(summary: Mapping[str, Any], destination: Path) -> Dict[str, Any]:
    assets_dir = destination / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    rendered: Dict[str, Any] = {key: value for key, value in summary.items() if key != "sections"}
    rendered_sections = []
    for section in summary.get("sections", ()):
        rendered_section = {key: value for key, value in section.items() if key != "cases"}
        rendered_cases = []
        for case in section.get("cases", ()):
            rendered_case = {key: value for key, value in case.items() if key != "_assets_source"}
            source = case.get("_assets_source", {})
            assets: Dict[str, str] = {}
            if isinstance(source, Mapping):
                for name, image in source.items():
                    filename = f"{_safe_name(section.get('id', 'section'))}_{_safe_name(case.get('id', 'case'))}_{_safe_name(name)}.png"
                    _save_asset(assets_dir / filename, str(name), np.asarray(image, dtype=np.float64))
                    assets[str(name)] = f"assets/{filename}"
            rendered_case["assets"] = assets
            rendered_cases.append(rendered_case)
        rendered_section["cases"] = rendered_cases
        rendered_sections.append(rendered_section)
    rendered["sections"] = rendered_sections
    return rendered


def _save_asset(path: Path, name: str, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        output = _to_uint8_rgb(image)
    elif image.ndim == 2:
        if any(token in name for token in ("raw", "radiance", "luma")):
            output = _to_uint8_gray(image)
        else:
            output = _to_uint8_heatmap(image)
    elif image.ndim == 1:
        output = _to_uint8_heatmap(np.repeat(image[:, None], 16, axis=1))
    else:
        raise ValueError(f"unsupported asset shape for {name}: {image.shape}")
    Image.fromarray(output).save(path)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)[:, :, :3]
    finite = np.isfinite(rgb)
    if not bool(np.any(finite)):
        return np.zeros(rgb.shape, dtype=np.uint8)
    high = float(np.nanpercentile(rgb[finite], 99.5))
    scale = 1.0 if high <= 1.0 else max(high, 1.0e-12)
    return np.round(np.clip(rgb / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    normalized = _normalize_display(image)
    return np.round(normalized * 255.0).astype(np.uint8)


def _to_uint8_heatmap(image: np.ndarray) -> np.ndarray:
    value = _normalize_display(image)
    anchors = np.asarray(
        [
            [18, 42, 66],
            [30, 126, 140],
            [98, 174, 112],
            [240, 190, 62],
            [196, 61, 51],
        ],
        dtype=np.float64,
    )
    position = value * float(len(anchors) - 1)
    low = np.floor(position).astype(int)
    high = np.clip(low + 1, 0, len(anchors) - 1)
    fraction = (position - low)[:, :, None]
    color = anchors[low] * (1.0 - fraction) + anchors[high] * fraction
    return np.round(color).astype(np.uint8)


def _normalize_display(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return np.zeros(arr.shape, dtype=np.float64)
    low = float(np.nanpercentile(arr[finite], 1.0))
    high = float(np.nanpercentile(arr[finite], 99.0))
    if high <= low:
        return np.clip(arr, 0.0, 1.0)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _render_html(summary: Mapping[str, Any]) -> str:
    tabs = [
        ("overview", "Overview"),
        ("hdr", "HDR"),
        ("metadata", "Sensor Metadata"),
        ("calibration-optics", "Calibration & Optics"),
        ("temporal", "Temporal"),
        ("dnn-contract", "DNN Contract"),
        ("camerae2e", "CameraE2E"),
    ]
    buttons = "".join(
        f'<button class="tab-button{" active" if index == 0 else ""}" type="button" role="tab" aria-selected="{"true" if index == 0 else "false"}" data-tab="{html_lib.escape(identifier)}">{html_lib.escape(title)}</button>'
        for index, (identifier, title) in enumerate(tabs)
    )
    sections = {str(row["id"]): row for row in summary.get("sections", ())}
    panels = [
        _render_overview(summary),
        _render_section(sections["hdr"]),
        _render_section(sections["metadata"]),
        _render_combined_section(
            "calibration-optics",
            "Calibration & Optics",
            (sections["calibration"], sections["cfa-optics"]),
        ),
        _render_section(sections["temporal"]),
        _render_section(sections["dnn-contract"]),
        _render_section(sections["camerae2e"]),
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_lib.escape(str(summary.get('title', 'PerceptionISP Example Suite')))}</title>
  <style>
    :root {{ color-scheme: light; --ink: #17212b; --muted: #5d6873; --line: #d9e0e5; --surface: #ffffff; --band: #f3f6f7; --accent: #176b72; --pass: #157347; --fail: #b42318; --skip: #8a5a13; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--surface); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }}
    header {{ padding: 30px max(24px, calc((100vw - 1440px) / 2)); border-bottom: 1px solid var(--line); background: var(--band); }}
    h1 {{ margin: 0 0 8px; font-size: 30px; line-height: 1.2; }}
    h2 {{ margin: 0 0 14px; font-size: 23px; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; }}
    p {{ line-height: 1.55; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .tabs {{ display: flex; overflow-x: auto; border-bottom: 1px solid var(--line); padding: 0 max(24px, calc((100vw - 1440px) / 2)); background: #fff; position: sticky; top: 0; z-index: 2; }}
    .tab-button {{ border: 0; border-bottom: 3px solid transparent; background: transparent; color: var(--muted); padding: 14px 16px 12px; font-weight: 650; white-space: nowrap; cursor: pointer; }}
    .tab-button.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 26px 24px 48px; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .status {{ display: inline-block; padding: 3px 7px; border: 1px solid currentColor; border-radius: 4px; font-size: 12px; font-weight: 750; text-transform: uppercase; }}
    .status.pass {{ color: var(--pass); }} .status.fail, .status.error {{ color: var(--fail); }} .status.skip, .status.not_run {{ color: var(--skip); }}
    .summary-band {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--line); border-radius: 6px; margin: 18px 0 24px; }}
    .summary-band div {{ padding: 14px; border-right: 1px solid var(--line); }} .summary-band div:last-child {{ border-right: 0; }}
    .summary-band strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    .note {{ border-left: 4px solid var(--accent); padding: 10px 14px; background: #edf5f5; margin: 16px 0; }}
    .table-scroll {{ width: 100%; max-width: 100%; overflow-x: auto; margin: 14px 0 26px; }}
    table {{ width: 100%; min-width: 680px; border-collapse: collapse; margin: 0; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }} th {{ background: var(--band); }}
    .case {{ border: 1px solid var(--line); border-radius: 6px; padding: 16px; margin: 16px 0; }}
    .section-block + .section-block {{ border-top: 1px solid var(--line); margin-top: 34px; padding-top: 28px; }}
    .case-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
    .assets {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 14px; }}
    figure {{ margin: 0; min-width: 0; }} figure img {{ display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #101820; border: 1px solid var(--line); }} figcaption {{ color: var(--muted); font-size: 12px; margin-top: 4px; overflow-wrap: anywhere; }}
    details {{ margin-top: 12px; }} pre {{ overflow: auto; max-height: 340px; padding: 12px; background: var(--band); border: 1px solid var(--line); border-radius: 4px; font-size: 12px; }}
    ul {{ line-height: 1.55; }}
    @media (max-width: 760px) {{ header, main {{ padding-left: 14px; padding-right: 14px; }} .tabs {{ padding-left: 4px; padding-right: 4px; }} .summary-band {{ grid-template-columns: 1fr 1fr; }} .summary-band div:nth-child(2) {{ border-right: 0; }} .case-head {{ display: block; }} }}
  </style>
</head>
<body>
  <header><h1>{html_lib.escape(str(summary.get('title', '')))}</h1><p>{html_lib.escape(str(summary.get('interpretation', '')))}</p></header>
  <nav class="tabs" role="tablist">{buttons}</nav>
  <main>{''.join(panels)}</main>
  <script>
    const buttons = Array.from(document.querySelectorAll('.tab-button'));
    const panels = Array.from(document.querySelectorAll('.tab-panel'));
    function loadPanel(id) {{
      const panel = panels.find((candidate) => candidate.dataset.panel === id);
      if (!panel) return;
      panel.querySelectorAll('img[data-src]').forEach((image) => {{ image.src = image.dataset.src; image.removeAttribute('data-src'); }});
    }}
    function activate(id) {{
      buttons.forEach((button) => {{ const active = button.dataset.tab === id; button.classList.toggle('active', active); button.setAttribute('aria-selected', String(active)); }});
      panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === id));
      loadPanel(id);
      history.replaceState(null, '', '#' + id);
    }}
    buttons.forEach((button) => button.addEventListener('click', () => activate(button.dataset.tab)));
    const requested = location.hash.slice(1); if (requested && panels.some((panel) => panel.dataset.panel === requested)) activate(requested); else loadPanel('overview');
  </script>
</body>
</html>
"""


def _render_overview(summary: Mapping[str, Any]) -> str:
    section_rows = "".join(
        "<tr>"
        f"<td>{html_lib.escape(str(row.get('title', '')))}</td>"
        f"<td>{_status_html(row.get('status'))}</td>"
        f"<td>{len(row.get('checks', ()))}</td>"
        f"<td>{len(row.get('cases', ()))}</td>"
        f"<td>{html_lib.escape(str(row.get('interpretation', '')))}</td>"
        "</tr>"
        for row in summary.get("sections", ())
    )
    limitations = "".join(f"<li>{html_lib.escape(str(value))}</li>" for value in summary.get("known_limitations", ()))
    return f"""
<section class="tab-panel active" data-panel="overview">
  <h2>Overview {_status_html(summary.get('status'))}</h2>
  <div class="summary-band"><div>Resolution<strong>{int(summary.get('width', 0))} x {int(summary.get('height', 0))}</strong></div><div>Seed<strong>{int(summary.get('seed', 0))}</strong></div><div>Selected groups<strong>{len(summary.get('selected_case_groups', ()))}</strong></div><div>CameraE2E required<strong>{html_lib.escape(str(summary.get('camerae2e_required', False)))}</strong></div></div>
  <div class="table-scroll"><table><thead><tr><th>Section</th><th>Status</th><th>Checks</th><th>Cases</th><th>Interpretation</th></tr></thead><tbody>{section_rows}</tbody></table></div>
  <h3>Known limitations</h3><ul>{limitations}</ul>
</section>"""


def _render_section(section: Mapping[str, Any]) -> str:
    return f"""
<section class="tab-panel" data-panel="{html_lib.escape(str(section.get('id', '')))}">
  {_render_section_body(section, heading_level=2)}
</section>"""


def _render_combined_section(panel_id: str, title: str, sections: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(section.get("status", "not_run")) for section in sections}
    if statuses.intersection({"fail", "error"}):
        status = "fail"
    elif statuses.intersection({"pass"}):
        status = "pass"
    elif statuses == {"skip"}:
        status = "skip"
    else:
        status = "not_run"
    blocks = "".join(
        f'<div class="section-block">{_render_section_body(section, heading_level=3)}</div>' for section in sections
    )
    return f"""
<section class="tab-panel" data-panel="{html_lib.escape(panel_id)}">
  <h2>{html_lib.escape(title)} {_status_html(status)}</h2>
  {blocks}
</section>"""


def _render_section_body(section: Mapping[str, Any], *, heading_level: int) -> str:
    title_tag = f"h{heading_level}"
    detail_tag = f"h{min(heading_level + 1, 6)}"
    checks = "".join(
        "<tr>"
        f"<td>{_status_html(row.get('status'))}</td>"
        f"<td><code>{html_lib.escape(str(row.get('id', '')))}</code></td>"
        f"<td>{html_lib.escape(str(row.get('description', '')))}</td>"
        f"<td>{html_lib.escape(str(row.get('evidence', '')))}</td>"
        "</tr>"
        for row in section.get("checks", ())
    )
    support = ""
    if section.get("support_matrix"):
        rows = "".join(
            f"<tr><td><code>{html_lib.escape(str(row.get('field', '')))}</code></td><td>{html_lib.escape(str(row.get('status', '')))}</td><td>{html_lib.escape(str(row.get('behavior', '')))}</td></tr>"
            for row in section.get("support_matrix", ())
        )
        support = f"<{detail_tag}>Support matrix</{detail_tag}><div class='table-scroll'><table><thead><tr><th>Field</th><th>Status</th><th>Behavior</th></tr></thead><tbody>{rows}</tbody></table></div>"
    notes = "".join(f"<li>{html_lib.escape(str(value))}</li>" for value in section.get("notes", ()))
    cases = "".join(_render_case(row) for row in section.get("cases", ()))
    empty = "<p>No cases were produced for this section.</p>" if not cases else ""
    return f"""
  <{title_tag}>{html_lib.escape(str(section.get('title', '')))} {_status_html(section.get('status'))}</{title_tag}>
  <div class="note">{html_lib.escape(str(section.get('interpretation', '')))}</div>
  {f'<ul>{notes}</ul>' if notes else ''}
  {support}
  <{detail_tag}>Checks</{detail_tag}>
  <div class="table-scroll"><table><thead><tr><th>Status</th><th>Check</th><th>Expectation</th><th>Evidence</th></tr></thead><tbody>{checks}</tbody></table></div>
  <{detail_tag}>Cases</{detail_tag}>{cases}{empty}"""


def _render_case(case: Mapping[str, Any]) -> str:
    assets = "".join(
        f'<figure><img data-src="{html_lib.escape(str(path))}" alt="{html_lib.escape(str(name))}" loading="lazy" decoding="async"><figcaption>{html_lib.escape(str(name).replace("_", " "))}</figcaption></figure>'
        for name, path in case.get("assets", {}).items()
    )
    details = json.dumps(json_ready({"metrics": case.get("metrics", {}), "metadata": case.get("metadata", {}), "warnings": case.get("warnings", [])}), indent=2)
    return f"""
<article class="case">
  <div class="case-head"><div><h3>{html_lib.escape(str(case.get('id', '')))}</h3><p>{html_lib.escape(str(case.get('description', '')))}</p></div>{_status_html(case.get('status'))}</div>
  <div class="assets">{assets}</div>
  <details><summary>Metrics and metadata</summary><pre>{html_lib.escape(details)}</pre></details>
</article>"""


def _status_html(value: Any) -> str:
    status = str(value or "unknown").lower()
    return f'<span class="status {html_lib.escape(status)}">{html_lib.escape(status)}</span>'


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "asset"


if __name__ == "__main__":
    raise SystemExit(main())
