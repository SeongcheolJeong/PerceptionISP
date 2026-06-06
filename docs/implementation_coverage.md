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
| DNN Export | RGB+aux six-channel tensor export, labels, manifest, PyTorch dataset adapter |
| Training Smoke | tiny PyTorch RGB+aux stem and optimization loop for path validation |
| Runtime Controller | rule-based HDR/noise/fast-path suggestions |
| Safety Monitor | exposure, visibility, focus, tint, DNN input validity |

Known limits:

- The Bayer demosaic block is an edge-aware numpy reference, not a production ISP demosaic.
- The RGB+aux training loop is a smoke test, not a real detector training recipe.
- No trained aux-aware detector checkpoint is included yet.
- CameraE2E integration is optional and environment-dependent.
- Latency values are engineering estimates, not measured hardware evidence.
