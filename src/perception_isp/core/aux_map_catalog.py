"""Machine-readable descriptions of every public PerceptionISP auxiliary map."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class AuxMapSpec:
    name: str
    group: str
    stage: str
    purpose: str
    expected_effect: str
    algorithm: str
    value_semantics: str
    derivation: str
    consumers: tuple[str, ...]
    applicability: str
    limitations: str
    rendering: str
    implementation_ref: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _spec(
    name: str,
    group: str,
    stage: str,
    purpose: str,
    expected_effect: str,
    algorithm: str,
    value_semantics: str,
    derivation: str,
    consumers: tuple[str, ...],
    applicability: str,
    limitations: str,
    rendering: str,
    implementation_ref: str,
) -> AuxMapSpec:
    return AuxMapSpec(
        name=name,
        group=group,
        stage=stage,
        purpose=purpose,
        expected_effect=expected_effect,
        algorithm=algorithm,
        value_semantics=value_semantics,
        derivation=derivation,
        consumers=consumers,
        applicability=applicability,
        limitations=limitations,
        rendering=rendering,
        implementation_ref=implementation_ref,
    )


AUX_MAP_SPECS: tuple[AuxMapSpec, ...] = (
    _spec("lens_gain", "normalization_calibration", "raw_physical_normalization", "Expose spatial lens-shading compensation and its noise amplification context.", "Lets downstream gating distinguish center and vignetted regions when a calibrated gain map exists.", "Nearest-resize the calibration lens_shading_gain map and multiply it into gain-corrected RAW.", "Positive gain; 1 means neutral and values above 1 amplify signal and modeled noise.", "calibration_passthrough", ("accurate", "dnn_extended", "noise_model"), "Meaningful only with a sensor/lens calibration map.", "A uniform value of 1 is missing/neutral calibration, not proof of uniform optics.", "sequential", "PerceptionISPPipeline._raw_physical_normalization"),
    _spec("defect_confidence", "normalization_calibration", "raw_physical_normalization", "Mark pixels replaced from the defect table.", "Allows DNN and health logic to reduce trust in interpolated sensor samples.", "Initialize ones and set calibrated defect coordinates to zero after local box-filter replacement.", "[0,1]; 0 marks a corrected defect and 1 an unlisted pixel.", "calibration_derived", (), "Requires an accurate defect-pixel table.", "An all-one map means no table entries were supplied, not that the sensor has no defects.", "bounded", "PerceptionISPPipeline._raw_physical_normalization"),
    _spec("hdr_exposure_source", "hdr_exposure", "hdr_fusion", "Identify the exposure with the largest fusion weight at each pixel.", "Makes exposure switching auditable and can condition highlight/shadow processing.", "Argmax of SNR × non-saturation × low-signal × inter-plane-consistency weights; use the darkest fallback when every weight is zero.", "Categorical exposure-plane index 0..E-1.", "model_derived", ("accurate", "edge_packet"), "Useful for multi-exposure inputs; single exposure is uniformly index 0.", "It is the dominant contributor, not a hard statement that other planes were unused.", "categorical", "PerceptionISPPipeline._hdr_fusion"),
    _spec("saturation", "hdr_exposure", "hdr_fusion", "Locate pixels saturated in any input exposure.", "Supports highlight-risk gating and prevents saturated structure from receiving high edge confidence.", "Threshold every normalized exposure and reduce with logical any across the exposure axis.", "Binary [0,1]; 1 means at least one source plane crossed hdr_saturation_threshold.", "model_derived", ("accurate", "fast", "dnn", "health"), "Applies to single and multi-exposure RAW.", "It does not mean the fused HDR result is clipped; a shorter plane may still recover the pixel.", "bounded", "PerceptionISPPipeline._hdr_fusion"),
    _spec("clipping_distance", "hdr_exposure", "hdr_fusion", "Express remaining headroom below the saturation threshold.", "Provides a continuous highlight-risk signal beyond the binary saturation map.", "Normalize threshold minus the maximum normalized exposure by the threshold and clip to [0,1].", "[0,1]; 0 is at/above threshold and larger values have more headroom.", "model_derived", ("accurate", "dnn_extended"), "Applies to the configured pseudo-RAW scale.", "It is based on the brightest input plane, not fused-radiance headroom.", "bounded", "PerceptionISPPipeline._hdr_fusion"),
    _spec("hdr_confidence", "hdr_exposure", "hdr_fusion", "Summarize usable HDR fusion support.", "Allows output and DNN validity to fall where exposures are saturated, noisy, or inconsistent.", "Sum nonnegative fusion weights and divide by exposure count, then clip to [0,1].", "[0,1]; higher means stronger aggregate usable exposure support.", "model_derived", ("accurate", "dnn_extended", "health", "color_confidence"), "Applies to single and multi-exposure inputs.", "This is a heuristic confidence and is not calibrated probability or HDR quality ground truth.", "bounded", "PerceptionISPPipeline._hdr_fusion"),
    _spec("ghost_motion_artifact", "hdr_exposure", "hdr_fusion", "Expose inter-plane radiance disagreement that can create ghosting.", "Can suppress unreliable HDR edges and reveal motion/flicker merge risk.", "Compute per-pixel radiance standard deviation divided by mean absolute radiance, clipped to [0,1].", "[0,1]; higher means greater normalized disagreement.", "heuristic_derived", ("psf_edge_likelihood", "temporal_flicker"), "Informative only for two or more exposure planes.", "No registration or optical flow is used; motion, JPEG differences, AE errors, and noise are conflated.", "bounded", "PerceptionISPPipeline._hdr_fusion"),
    _spec("luma", "cfa_spectral", "cfa_decode", "Provide a dense structural intensity channel close to the sensor domain.", "Supports low-latency edge and temporal processing without waiting for display rendering.", "For Bayer use the measured/interpolated green plane; clear, IR, and mono layouts use their decoder-specific intensity.", "Nonnegative linear intensity on the fused pseudo-RAW scale.", "model_derived", ("fast", "raw_like", "temporal_state"), "Applicable to all supported CFA modes.", "Bayer green is a luma proxy, not colorimetric luminance.", "sequential", "PerceptionISPPipeline._cfa_pixel_structure_decoder"),
    _spec("clear_channel", "cfa_spectral", "cfa_decode", "Expose clear-pixel signal for clear-filter CFA layouts.", "May improve low-light structure when genuine clear pixels and calibration are available.", "Interpolate C samples for RCCB/RCCC/RCCG; use fused signal for mono; emit zeros for Bayer/RGB-IR.", "Nonnegative linear clear signal; zero when the active CFA has no clear samples.", "sensor_channel_derived", ("accurate_ir_or_clear",), "Only physically meaningful for native clear/mono CFA capture.", "CameraE2E Bayer and RGB proxy bridges do not create native clear measurements.", "sequential", "_decode_clear_cfa/_decode_monochrome"),
    _spec("ir_channel", "cfa_spectral", "cfa_decode", "Expose interpolated infrared-pixel signal.", "Can let a model reason about IR illumination and visible-color contamination.", "Interpolate IR samples in RGB-IR mode; emit zeros for Bayer, clear, and mono modes.", "Nonnegative linear IR signal; zero when the CFA has no IR samples.", "sensor_channel_derived", ("accurate_ir_or_clear", "ir_contamination"), "Only physically meaningful for native RGB-IR capture and crosstalk calibration.", "An RGB-derived CameraE2E proxy is not a native IR measurement.", "sequential", "_decode_rgb_ir"),
    _spec("demosaic_confidence", "cfa_spectral", "cfa_decode_edge", "Estimate reliability of reconstructed color samples.", "Supports artifact suppression and downweights uncertain color edges.", "Combine CFA neighborhood support, horizontal/vertical gradient separation and local contrast, then mix with edge confidence and non-edge support.", "[0,1]; higher indicates stronger reconstruction support.", "heuristic_derived", ("accurate", "dnn_extended", "image_formation"), "Decoder-specific; valid as a relative diagnostic for supported CFA modes.", "Not calibrated against demosaic ground truth and partly depends on later edge confidence.", "bounded", "_decode_bayer_edge_aware/PerceptionISPPipeline._edge_structure_engine"),
    _spec("color_confidence", "cfa_spectral", "color_spectral", "Estimate where color information is trustworthy.", "Can reduce reliance on chroma under low SNR, saturation, poor HDR support, or strong global imbalance.", "Multiply normalized SNR, non-saturation, global WB-balance confidence, and HDR confidence.", "[0,1]; higher means stronger modeled color reliability.", "heuristic_derived", ("accurate", "dnn_extended", "health"), "Applicable to decoded RGB but depends on calibration quality.", "Global channel imbalance is only a WB heuristic and can penalize legitimately colored scenes.", "bounded", "PerceptionISPPipeline._color_spectral_engine"),
    _spec("log_r_over_g", "cfa_spectral", "color_spectral", "Represent red chromaticity relative to green.", "Offers illumination/color cues with reduced dependence on overall brightness.", "Compute safe log(R/G) after the camera color matrix.", "Signed log ratio; 0 means R equals G.", "model_derived", (), "Applicable when decoded camera RGB is meaningful.", "Sensitive to color matrix, demosaic error, clipping, and very small green values.", "diverging", "PerceptionISPPipeline._color_spectral_engine"),
    _spec("log_b_over_g", "cfa_spectral", "color_spectral", "Represent blue chromaticity relative to green.", "Offers illumination/color cues with reduced dependence on overall brightness.", "Compute safe log(B/G) after the camera color matrix.", "Signed log ratio; 0 means B equals G.", "model_derived", (), "Applicable when decoded camera RGB is meaningful.", "Sensitive to color matrix, demosaic error, clipping, and very small green values.", "diverging", "PerceptionISPPipeline._color_spectral_engine"),
    _spec("ir_contamination", "cfa_spectral", "color_spectral", "Estimate the fraction of signal attributable to IR.", "Can gate visible-color features where IR leakage is strong.", "Compute IR divided by mean visible RGB plus IR.", "[0,1]; higher means larger modeled IR contribution.", "heuristic_derived", (), "Only meaningful for genuine RGB-IR input.", "Uniform zero on Bayer is not evidence of an IR-free physical sensor.", "bounded", "PerceptionISPPipeline._color_spectral_engine"),
    _spec("noise_variance", "noise_reliability", "noise_uncertainty", "Expose modeled per-pixel uncertainty in the fused RAW signal.", "Allows denoising, fast-path features, and DNN inputs to distinguish signal from noisy regions.", "Sum shot, read, temperature/exposure dark-current, quantization, and calibration-residual variance; scale by lens_gain squared.", "Positive modeled variance in normalized signal units squared.", "model_derived", ("accurate", "fast", "dnn_extended", "edge_engine"), "Requires representative sensor-noise and calibration coefficients.", "CameraE2E bridge defaults are not a calibrated nuScenes camera noise model.", "positive_robust", "PerceptionISPPipeline._noise_uncertainty_engine"),
    _spec("snr_map", "noise_reliability", "noise_uncertainty", "Provide bounded signal reliability.", "Selected as the compact reliability channel in the stable RGB+Aux tensor and as a gate for edge/color confidence.", "Compute signal/sqrt(variance), then map SNR to SNR/(SNR+1).", "[0,1]; higher means stronger modeled signal relative to noise.", "model_derived", ("accurate", "dnn", "edge_engine", "color_confidence"), "Relative validity follows the supplied noise model.", "It is not measured SNR and is optimistic or pessimistic when calibration coefficients are wrong.", "bounded", "PerceptionISPPipeline._noise_uncertainty_engine"),
    _spec("noise_normalized_gradient", "noise_reliability", "noise_uncertainty", "Measure local intensity change relative to expected noise.", "Helps distinguish structured transitions from noise-only gradients.", "Divide luma gradient magnitude by sqrt(2 × modeled variance).", "Nonnegative, unbounded standardized gradient-like score.", "model_derived", (), "Useful when both luma and the noise model are meaningful.", "Large texture and demosaic artifacts can also produce high values.", "positive_robust", "PerceptionISPPipeline._noise_uncertainty_engine"),
    _spec("edge_strength", "edge_optics", "edge_structure", "Represent local structural gradient magnitude.", "Supports boundary-aware image formation, sparse EdgePackets, and DNN spatial features.", "Build a noise-weighted luma/R-G/B-G structure tensor, take sqrt of the largest eigenvalue, and normalize by the frame maximum.", "[0,1] frame-relative strength.", "model_derived", ("accurate", "fast", "dnn", "edge_packet"), "Applicable to all decoded inputs.", "Frame-max normalization prevents absolute comparison across unrelated runs without stored statistics.", "bounded", "PerceptionISPPipeline._edge_structure_engine"),
    _spec("edge_orientation", "edge_optics", "edge_structure", "Expose the dominant local edge direction.", "Can support orientation-aware filtering or geometric feature extraction.", "Use half-angle atan2 of the structure-tensor off-diagonal and diagonal difference.", "Cyclic radians in approximately [-pi/2, pi/2].", "model_derived", ("edge_packet",), "Meaningful where edge strength/confidence is sufficient.", "Orientation is unstable in flat, noisy, or isotropic texture regions.", "cyclic", "PerceptionISPPipeline._edge_structure_engine"),
    _spec("edge_confidence", "edge_optics", "edge_structure", "Estimate whether a detected edge is structurally and sensor-wise reliable.", "Expected to suppress noisy, saturated, blurred, or isotropic edges while retaining supported boundaries.", "Multiply structure-tensor anisotropy, MTF, exp(-0.5×PSF sigma squared), non-saturation, and SNR.", "[0,1]; higher means stronger modeled edge reliability.", "heuristic_derived", ("accurate", "fast", "edge_packet", "image_formation"), "Depends on valid noise and optics calibration.", "A high value is not object-boundary probability or detector performance evidence.", "bounded", "PerceptionISPPipeline._edge_structure_engine"),
    _spec("edge_evidence", "edge_optics", "edge_structure", "Combine edge magnitude and reliability into one compact feature.", "Selected for DNN/proposal experiments where strength alone admits noise and confidence alone can be weak at strong boundaries.", "Robust-normalize strength and confidence with p1/p99 ranges and take their geometric mean.", "[0,1] frame-relative combined evidence.", "heuristic_derived", ("accurate", "fast", "dnn_extended"), "Applicable to decoded inputs with meaningful edge maps.", "Percentile normalization makes values run-relative rather than calibrated probabilities.", "bounded", "_combined_edge_evidence"),
    _spec("edge_type", "edge_optics", "edge_structure", "Separate luminance and chromatic edge evidence.", "Can distinguish intensity boundaries from color-only transitions and demosaic-sensitive locations.", "Normalize luma and maximum chroma gradient by frame maxima and classify at threshold 0.12 as none/luma/chroma/both.", "Categorical: 0 none, 1 luma, 2 chroma, 3 both.", "heuristic_derived", (), "Applicable when decoded chroma differences are meaningful.", "Fixed frame-relative thresholds are diagnostic and not universally calibrated.", "categorical", "PerceptionISPPipeline._edge_structure_engine"),
    _spec("blur_focus_confidence", "edge_optics", "edge_structure", "Summarize local focus support from strong reliable edges.", "Can flag regions where detail-sensitive downstream features should be downweighted.", "Average edge confidence with edge strength normalized by the configured high-percentile strength threshold.", "[0,1]; higher means stronger local focus-like support.", "heuristic_derived", ("accurate", "dnn_extended", "health"), "Relative within a frame and most meaningful around edges.", "Texture scarcity, motion blur, defocus, and low contrast are not separated.", "bounded", "_blur_focus_confidence"),
    _spec("mtf_confidence", "edge_optics", "edge_structure", "Carry calibrated spatial MTF reliability into edge processing.", "Allows known lens/sensor resolution loss to reduce edge confidence.", "Nearest-resize calibration.mtf_confidence_map, defaulting to one.", "Normally [0,1]; one means no attenuation in the model.", "calibration_passthrough", ("accurate", "edge_confidence"), "Meaningful only with measured/calibrated MTF data.", "Uniform one is a neutral default, not measured perfect MTF.", "bounded", "PerceptionISPPipeline._edge_structure_engine"),
    _spec("psf_sigma", "edge_optics", "edge_structure", "Expose calibrated local PSF blur scale.", "Lets downstream logic reason about spatially varying optical blur.", "Nearest-resize calibration.psf_sigma_map and clamp to nonnegative values.", "Nonnegative sigma in calibration-defined sensor-pixel units.", "calibration_passthrough", ("accurate", "psf_blur_confidence"), "Meaningful only with calibrated PSF data.", "Zero is the default when no PSF map is supplied and is not a measured diffraction-free result.", "positive_robust", "PerceptionISPPipeline._edge_structure_engine"),
    _spec("psf_blur_confidence", "edge_optics", "edge_structure", "Convert PSF width into a bounded sharpness prior.", "Expected to reduce trust in edges where calibrated blur is large.", "Compute exp(-0.5 × psf_sigma squared).", "[0,1]; 1 at sigma 0 and decreasing with blur scale.", "calibration_derived", ("accurate", "dnn_extended", "edge_confidence"), "Depends entirely on PSF calibration.", "A uniform one usually reflects missing/default PSF calibration.", "bounded", "_psf_blur_confidence"),
    _spec("psf_edge_likelihood", "edge_optics", "edge_structure", "Prefer reliable edges that are not explained by HDR disagreement.", "Selected as a compact edge/optics cue for DNN experiments under blur and multi-exposure stress.", "Subtract ghost_motion_artifact from edge_confidence and clip to [0,1].", "[0,1]; higher means confident edge with low inter-plane disagreement.", "heuristic_derived", ("accurate", "dnn_extended"), "Useful as a diagnostic fusion of edge and HDR consistency.", "It is not a fitted PSF likelihood despite the historical name.", "bounded", "PerceptionISPPipeline._edge_structure_engine"),
    _spec("rolling_row_fraction", "timing_temporal", "optics_geometry_timing", "Expose normalized sensor-row readout position.", "Allows models or diagnostics to condition on rolling-readout timing.", "Generate a linear 0..1 vector over rows and reverse it for bottom-to-top readout.", "One-dimensional [0,1] row coordinate.", "metadata_derived", (), "Requires correct readout direction; row timestamps are available in Trace.", "No rolling-shutter geometric compensation is performed.", "bounded_strip", "PerceptionISPPipeline._optics_geometry_timing_engine"),
    _spec("temporal_difference", "timing_temporal", "temporal_flicker", "Expose change from the previous processed frame.", "Supports low-latency motion/change awareness and frozen-frame diagnostics.", "Absolute difference between current linear perception luma and PreviousFrameState luma; zero without compatible state.", "Nonnegative linear intensity difference.", "temporal_derived", ("fast",), "Requires correctly chained same-camera PreviousFrameState.", "No motion compensation; ego/object motion, AE, JPEG artifacts, and flicker are mixed.", "temporal_shared", "PerceptionISPPipeline._led_flicker_temporal_engine"),
    _spec("temporal_consistency", "timing_temporal", "temporal_flicker", "Convert frame change into bounded stability.", "Allows downstream paths to downweight temporally unstable regions.", "Compute exp(-temporal_difference / temporal_flicker_threshold).", "[0,1]; 1 is unchanged and lower values indicate larger change.", "temporal_derived", ("edge_packet", "flicker"), "Requires compatible previous-frame state.", "It measures photometric consistency, not correspondence or optical flow confidence.", "bounded", "PerceptionISPPipeline._led_flicker_temporal_engine"),
    _spec("led_flicker_confidence", "timing_temporal", "temporal_flicker", "Highlight bright temporally unstable or HDR-disagreeing regions.", "Intended to expose LED/flicker risk to runtime control and downstream gating.", "Combine (1-temporal_consistency) × bright-light gate with HDR disagreement × saturation, then clip.", "[0,1]; higher means stronger flicker-like evidence.", "heuristic_derived", ("runtime_control",), "Most informative for bright light sources in a chained sequence or multi-exposure input.", "Not an LED classifier; motion, exposure change, and compression can raise it.", "bounded", "PerceptionISPPipeline._led_flicker_temporal_engine"),
    _spec("light_source_confidence", "timing_temporal", "temporal_flicker", "Represent bright light-source-like regions with a flicker penalty.", "Can help distinguish stable lights from unstable highlights.", "Map luma above 0.65 into a 0..1 bright gate and multiply by 1-0.5×flicker.", "[0,1]; higher means bright and relatively stable.", "heuristic_derived", (), "Diagnostic for the configured linear intensity scale.", "Brightness alone cannot distinguish lamps, reflections, sky, or saturated surfaces.", "bounded", "PerceptionISPPipeline._led_flicker_temporal_engine"),
)

AUX_MAP_CATALOG: Mapping[str, AuxMapSpec] = {spec.name: spec for spec in AUX_MAP_SPECS}


def validate_aux_map_catalog(map_names: Iterable[str]) -> None:
    actual = {str(name) for name in map_names}
    expected = set(AUX_MAP_CATALOG)
    missing = sorted(actual - expected)
    stale = sorted(expected - actual)
    if missing or stale:
        raise ValueError(
            "Aux map catalog does not match pipeline maps: "
            f"uncatalogued={missing}, not_produced={stale}"
        )


def aux_map_catalog_json() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in AUX_MAP_SPECS]


__all__ = [
    "AUX_MAP_CATALOG",
    "AUX_MAP_SPECS",
    "AuxMapSpec",
    "aux_map_catalog_json",
    "validate_aux_map_catalog",
]
