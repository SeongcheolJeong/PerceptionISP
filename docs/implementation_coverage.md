# Implementation Coverage

This implementation is deliberately broad rather than hardware-optimized.

| Architecture Block | Software implementation |
| --- | --- |
| Sensor Interface / Metadata | `RawFrame`, `SensorMetadata`, RAW shape validation, line timing metadata |
| Calibration Loader | `CalibrationProfile`, model/unit/runtime-compatible fields |
| RAW Physical Normalization | decompand, black, FPN/DSNU, PRNU, defect pixels, lens gain |
| Noise / Uncertainty | shot/read/dark/quantization/calibration variance, SNR, noise-normalized gradient |
| CFA Decoder | RGGB/BGGR/GRBG/GBRG, RCCB/RCCC/RCCG, RGB-IR, mono/thermal |
| HDR Fusion | multi-exposure radiance normalization, saturation/source/confidence/ghost maps |
| Edge / Structure | CFA-aware structure tensor, orientation, confidence, edge type, MTF/focus maps |
| Color / Spectral | camera RGB, perception CCM, log R/G, log B/G, color confidence, IR contamination |
| Geometry / Timing | row timestamps, rolling fraction, intrinsic/extrinsic metadata, optional radial dewarp |
| LED / Temporal | temporal difference, consistency, flicker confidence, light-source confidence |
| Task Image Formation | human RGB, vision RGB, raw-like tensor |
| Output Formatter | accurate full tensor, fast stripe tensor, sparse edge packets |
| Runtime Controller | rule-based HDR/noise/fast-path suggestions |
| Safety Monitor | exposure, visibility, focus, tint, DNN input validity |

Known limits:

- The demosaic and dewarp blocks are intentionally simple numpy baselines.
- No learned DNN branch is included yet; tensors are prepared for downstream models.
- CameraE2E integration is optional and environment-dependent.
- Latency values are engineering estimates, not measured hardware evidence.
