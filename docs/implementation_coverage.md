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
| Evidence / Claiming | synthetic mechanism validation, CFA stress sweep, edge-confidence suite, object edge-fidelity suite, scene-information stress suite, aux contribution audit, CFA/LensPSF detector sweep, CFA/LensPSF proposal audit, CFA/LensPSF native-CFA audit, paired-bootstrap claim gates, broad-superiority and FP-reducer profiles, task-group metrics, task gate, condition-specific metrics, condition robustness gate, RGB+aux training rollup, benchmark-protocol coverage checklist, claim-readiness dashboard, one-shot readiness orchestration |
| Runtime Controller | rule-based HDR/noise/fast-path suggestions |
| Safety Monitor | exposure, visibility, focus, tint, DNN input validity |

Known limits:

- The Bayer demosaic block is an edge-aware numpy reference, not a production ISP demosaic.
- The RGB+aux compact dense detector is still a learning-path benchmark, not a claim-quality detector.
- The RGB+aux smoke checkpoint predicts one generic box and is not a useful trained detector.
- The current extended RGB+aux tensor is 15 channels after adding PSF blur and
  PSF edge-likelihood aux channels. Older 13-channel export/train/eval artifacts
  remain historical diagnostics; direct dense metrics are still weak and should
  not be used as a claim-quality detector result. A current 15-channel
  CameraE2E-backed smoke artifact exists at
  `reports/perception_rgb_aux_15ch_camerae2e_grbg_smoke_rollup/index.html`; it
  verifies export/load/train/checkpoint only, not detector performance.
- The task gate currently fails the recall-improvement profile on the KITTI
  t001 evidence bundle, so task-level recall improvement should not be claimed.
- Synthetic mechanism validation shows expected low-light, glare, low-MTF, and
  CFA-support map behavior, but it is not a substitute for real adverse
  condition datasets or downstream detector performance.
- The CFA stress sweep ranks synthetic front-end signals by CFA and condition;
  it is diagnostic evidence, not a product sensor or detector-performance
  claim.
- The edge-confidence suite validates difficult-edge confidence behavior under
  synthetic low-light, glare, and low-MTF stress, but it remains front-end
  signal evidence rather than detector-performance evidence.
- The object edge-fidelity suite compares object/sensor edge oracles against
  HumanISP RGB edges, PerceptionISP RGB edges, and aux edge maps across CFA and
  LensPSF blur, but it remains front-end edge evidence rather than
  detector-performance evidence. PSF effects can be nearly invisible when the
  PSF footprint is much smaller than the sensor pixel pitch or is applied only
  after low-resolution sampling. `psf_sigma_map` now feeds PSF blur-confidence
  and PSF edge-likelihood aux maps when calibration provides it.
- `eval_cli` now supports `--psf-sigma`, which injects a constant
  `psf_sigma_map` into RAW calibration and records it in run config, sample
  metadata, and RAW provenance. `cfa_lenspsf_detector_sweep` uses that path to
  run fixed-detector CFA/LensPSF condition sweeps and report remap/provenance
  checks. The current `native_bayer_v1` KITTI val32 sweep covers
  GRBG/RGGB/BGGR/GBRG x PSF `0.0/0.8/1.6` at `640 x 192`, with 384/384 samples
  recorded as true native CFA mosaics and no bridge remapping. It is diagnostic
  condition evidence, not broad detector robustness.
- `cfa_lenspsf_proposal_audit` joins the KITTI val32 CFA/LensPSF detector sweep
  to same-sample proposal edge and source-scene-edge correlations. The current
  native report removes 151 FP and 0 TP proposals across 12 conditions, with
  source-scene-edge support directionally positive in 12/12 conditions and
  aux-edge support positive in 9/12. This is calibrated proposal-path bridge
  evidence, not incremental aux-only ablation and not a trained RGB+aux DNN
  result.
- `cfa_lenspsf_native_audit` separates native CameraE2E source-CFA rows from
  rows that were remapped to a requested target CFA. The current
  `native_bayer_v1` KITTI val32 audit has 12 native rows with 384 samples and 0
  remapped rows for `RGGB`, `GRBG`, `BGGR`, and `GBRG`. The older val32
  `bayer_psf` report predates `native_bayer_v1`; its non-GRBG rows remain
  historical bridge/remap sensitivity evidence only.
- The scene edge-confidence suite compares HumanISP RGB edge proxies,
  PerceptionISP RGB edge proxies, aux edge strength, and aux edge confidence
  against a higher-resolution real-scene edge proxy after CameraE2E sensor
  sampling. It is scene-edge front-end evidence, not object-boundary or
  detector-performance evidence.
- The scene-information stress suite validates high-resolution scene detail
  loss, CFA chroma alias/color uncertainty, and sub-pixel signal fill-factor
  loss, but it is diagnostic scene-to-sensor evidence and does not show that
  PerceptionISP can recover information absent from RAW.
- The aux contribution audit shows aux features can add proposal-scoring FP
  reduction beyond score/label calibration, but it is detector-side calibration
  evidence rather than proof of a trained RGB+aux DNN. The same-sample bridge
  now also reports proposal-level edge support correlation: low edge support
  identifies removed FP proposals versus kept TP proposals with AUC `0.6904`,
  and source scene-edge support inside the same proposal boxes gives AUC
  `0.6681`.
- The success/failure casebook renders representative sample-level wins and
  counterexamples from the same comparison report used by claim gates. The
  current KITTI val1496 casebook selects 32 visual cases and shows
  `fp_reduction_success=304`, `recall_tradeoff=24`, `recall_loss_failure=56`,
  and `fp_regression_failure=57` across the full run. It is qualitative review
  evidence, not a replacement for metric gates or native RAW/CFA coverage.
- The condition gate currently uses KITTI metadata/proxy slices; it does not replace real night/rain/fog/HDR RAW datasets.
- The benchmark-protocol checklist is a claim blocker/coverage tool; it does not create missing real-RAW or adverse-condition evidence by itself.
- CameraE2E integration is optional and environment-dependent.
- Latency values are engineering estimates, not measured hardware evidence.
