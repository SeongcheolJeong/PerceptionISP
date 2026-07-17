# Evidence and Limitations

## Current Engineering Evidence

The repository implements and tests the end-to-end contracts needed for
feasibility work: native/simulated RAW input, CFA-aware ISP processing,
HumanISP and PerceptionISP outputs, Aux export, learned RGB+Aux input, task
evaluation, condition sweeps, and report generation.

Important observed results from the retained v0.1 experiment history are:

| Experiment | Observation | Interpretation |
| --- | --- | --- |
| KITTI val128 compact DNN, 12 epochs | `dR50=+0.0701`, `dSmallR50=+0.0333`, `dFP50=-36.4375` | Positive pilot signal, but the quality gate failed and the eval split was small. |
| KITTI val512 compact DNN, 12 epochs | `dP50=-0.0049`, `dR50=-0.0067`, `dSmallR50=-0.0243`, `dFP50=+19.6797` | Scale-up counterexample; current learned model does not robustly beat RGB-only. |
| Native CFA/LensPSF sweep | 12 conditions, 1,536 samples, source CFA equals target CFA, zero remap | Valid condition-sensitivity evidence, not broad task superiority. |
| Incremental Aux calibration ablation | recall wins in 12/12 conditions, mean `dR50=+0.0034`; FP wins in 0/12, mean `dFP50=+0.0540` | Recall/FP tradeoff requiring a better learned operating point. |
| KITTI box-boundary proxy | Aux edge-confidence mean `dF1=+0.0014`, win rate `0.5554`; stronger small-box diagnostic | Edge signal is present, but boxes are not contour ground truth. |

The detailed historical tables remain in
`docs/history/evaluation_status_v0.1.md`.

The executable `perception-isp example suite` adds deterministic mechanism
checks for HDR radiance recovery, metadata/gain consistency, calibration,
CFA/PSF, temporal timing, and the 6/16-channel DNN input contract. These checks
show that the software blocks respond in controlled directions. They do not
replace a trained, held-out perception benchmark.

The data/simulator-backed examples add four narrower contract checks:

- `example camerae2e-hdr` verifies explicit integration vectors or an
  AE-anchored same-scene bracket, shared voltage scaling, sensor-noise seed
  readback, and the analog-voltage-to-pseudo-code boundary. Its RGB source
  still makes the output pseudo-RAW rather than native ADC data;
- `example nuscenes-temporal` drives one stateful CameraE2E AE sequence over
  processed JPEG frames and verifies control delay, temporal ISP state, and
  camera/ego metadata, not native RAW or an HDR bracket;
- `example nuscenes-interframe-hdr-stress` uses stateful AE plus an explicit
  `0/-2/+2 EV` AEB cycle over consecutive same-camera JPEGs and forms
  unregistered inter-frame `E x H x W` groups. It is a
  motion/flicker/merge-risk stress test, not calibrated or simultaneous HDR;
- `example native-hdr` accepts only a strict multi-exposure NPZ+JSON bundle
  attested as native sensor data and rejects pseudo-RAW by default. Native
  format eligibility is reported separately from HDR-quality eligibility,
  which additionally requires verified source origins for exposure, gain,
  CFA, and black/white calibration.

These examples do not perform optical-flow/pose-based registration. Their
motion and ghost maps expose disagreement risk; they do not demonstrate
deghosted HDR reconstruction.

`report showcase` composes these contracts into one fresh, auditable run over
six local nuScenes RGB frames. It creates 11 PerceptionISP results and all 33
Aux maps per result (363 map artifacts), plus opt-in intermediate trace arrays.
The report records zero cache hits and refuses CameraE2E/synthetic fallback.
Its Aux catalog explains algorithm, expected use, value semantics,
applicability, and limitations; expected effects are design intent rather than
measured downstream task gains.

The evidence levels must not be collapsed: static CameraE2E HDR re-captures one
fixed RGB scene, temporal nuScenes keeps separate AE-controlled single-exposure
frames, inter-frame nuScenes intentionally groups AEB slots from different
moments without registration, and native HDR consumes sensor-provided exposure
planes. Only the last is
native-source evidence, and even it needs separate alignment/deghosting tests
for dynamic-scene claims.

The source nuScenes dataset and JPEGs are user-provided local data and are not
bundled here. A deliberately selected public report snapshot may include
derived PNG previews only when it carries a report-local data notice,
attribution, CC BY-NC-SA 4.0, and the additional nuScenes Dataset Terms. Camera
`sample_data` is 12 Hz, while key samples and annotations are 2 Hz; non-key
camera frames must not be treated as independently annotated samples.

Metadata provenance separates source values from simulated or assumed values
without extending `SensorMetadata`. The flat `metadata_field_origins` map uses
`source_dataset`, `simulator_configured`, `simulator_readback`,
`bridge_assumed`, and `unknown`; converter origins are kept separately from
nuScenes source metadata.

The reports expose field/value/origin, AE control, and inter-frame exposure-plane
source tables. AE/AEB CameraE2E calls use distinct deterministic seeds so every
frame does not replay seed 0. This is not a calibrated temporal
noise model: CameraE2E uses one RNG for shot/read/FPN, so fixed-pattern noise is
not held constant across independent calls. Likewise, RCCB/RGBIR targets are
bridge remosaic proxies from RGB, not native clear/IR sensor measurements.

## Claims Supported Today

- A software PerceptionISP can generate stable RGB and auxiliary evidence from
  the same RAW frame used by a HumanISP baseline.
- Source CFA, target CFA, LensPSF, and remosaic provenance can be audited.
- Aux edge evidence contains information correlated with difficult object and
  boundary cases.
- RGB+Aux early-fusion training, feature distillation, held-out evaluation, and
  counterexample analysis are technically executable.
- Post-hoc Aux calibration can change the recall/FP operating point while
  preserving RGB detector class labels.

## Claims Not Supported Today

- PerceptionISP generally outperforms HumanISP for object detection.
- Aux maps consistently improve a pretrained DNN across datasets and seeds.
- The current software blocks match production automotive ISP quality,
  throughput, power, or fixed-point behavior.
- Box-edge proxy metrics prove true object-contour accuracy.
- Synthetic RGB-to-RAW remosaic results are equivalent to real native RAW.
- CameraE2E can recover source-camera RAW or clipped radiance from a rendered
  RGB/JPEG input.
- CameraE2E `sensor.volts` mapped to `[64, 4095]` is equivalent to source-camera
  ADC output or a standards-compliant DNG.
- Adjacent nuScenes JPEG frames form a radiometrically calibrated HDR bracket.
- A passing nuScenes inter-frame HDR stress report proves radiance recovery,
  registration, or deghosting.
- Preserving nuScenes poses as metadata means temporal frames were
  geometrically registered.
- A favorable threshold selected on validation proves a held-out improvement.

## Required Next Gates

1. Train matched RGB-only and gated RGB+Aux models on a larger native-RAW set
   using at least three seeds and identical budgets.
2. Pretrain the Aux stem with feature distillation, then test frozen-RGB and
   joint fine-tuning separately.
3. Evaluate small, thin/long, weak-edge, low-MTF, low-light, glare, and
   demosaic-artifact slices.
4. Use segmentation contour ground truth or renderer scene truth for boundary
   F1 rather than object boxes alone.
5. Repeat CFA/LensPSF sweeps with native pattern provenance and sufficiently
   resolved PSF kernels.
6. Report AP50/AP75, APs/APm/APl, recall, FP/sample, boundary F1, calibration,
   bootstrap intervals, and visual counterexamples together.

## Ground-Truth Bias

Most public detection/segmentation annotations are drawn on rendered RGB. This
can favor the appearance of that RGB pipeline. Mitigations are:

- preserve native RAW plus paired annotation when available;
- use scene geometry, renderer masks, or high-resolution source contours;
- evaluate several display/ISP renderings against one scene-level truth;
- report RGB-annotation results as task evidence, not sensor-truth proof.

## Decision Rule

A performance claim should pass only when the gain is held out, multi-seed,
statistically bounded, condition-relevant, and not paid for by an unacceptable
FP or latency increase. A counterexample must remain visible in the report.
