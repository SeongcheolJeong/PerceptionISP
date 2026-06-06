# Implementation Coverage

This implementation is deliberately broad rather than hardware-optimized.
See `docs/literature_basis.md` for the RAW/sensor-native perception evidence
boundary used by the benchmark protocol and claim gates.

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
| DNN Export | stable RGB+aux six-channel tensor, extended sensor-native aux tensor, labels, manifest, PyTorch dataset adapter |
| Training Smoke | tiny PyTorch RGB+aux stem, compact dense detector, channel ablations, checkpoint save/load, eval split, training/eval rollup |
| Learned Adapter | `RGBAuxTorchSmokeDetector` and `RGBAuxTorchDenseDetector` load checkpoints into the comparison harness |
| Evidence / Claiming | paired-bootstrap claim gates, broad-superiority and FP-reducer profiles, task-group metrics, condition-specific metrics, condition robustness gate, RGB+aux training rollup, benchmark-protocol coverage checklist, claim-readiness dashboard, one-shot readiness orchestration |
| Runtime Controller | rule-based HDR/noise/fast-path suggestions |
| Safety Monitor | exposure, visibility, focus, tint, DNN input validity |

Known limits:

- The Bayer demosaic block is an edge-aware numpy reference, not a production ISP demosaic.
- The RGB+aux compact dense detector is still a learning-path benchmark, not a claim-quality detector.
- The RGB+aux smoke checkpoint predicts one generic box and is not a useful trained detector.
- The extended 13-channel RGB+aux tensor has export/train/eval evidence, but
  the direct dense metrics remain diagnostic and weak.
- The condition gate currently uses KITTI metadata/proxy slices; it does not replace real night/rain/fog/HDR RAW datasets.
- The benchmark-protocol checklist is a claim blocker/coverage tool; it does not create missing real-RAW or adverse-condition evidence by itself.
- CameraE2E integration is optional and environment-dependent.
- Latency values are engineering estimates, not measured hardware evidence.
