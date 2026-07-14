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
