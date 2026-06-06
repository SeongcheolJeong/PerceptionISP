# Evaluation Status

This project now has an end-to-end perception comparison path:

1. Load or synthesize an RGB scene.
2. Convert the scene to RAW through CameraE2E when available.
3. Run the same RAW through the HumanISP-compatible RGB path and the PerceptionISP path.
4. Run detectors on Human RGB, Perception RGB, Perception RGB+aux fusion, and Perception auxiliary maps.
5. Compute detection metrics and write an HTML report with overlay images.

## What Is Implemented

- Synthetic labeled scenes for fast local regression.
- CameraE2E-backed synthetic scenes for simulator integration checks.
- Real-image smoke test with YOLO pseudo labels.
- YOLO-format dataset loader for converted KITTI/BDD/COCO-style subsets.
- Native KITTI `image_2` + `label_2` loader.
- Conservative RGB+aux fusion adapter that keeps RGB detector labels and adds
  aux-map support metadata from edge, saturation, and reliability channels.
- RGB+aux DNN export that writes six-channel tensors and labels for downstream
  training.
- Tiny PyTorch smoke training loop proving the exported six-channel tensors can
  be consumed by a DNN stem and optimized.
- HTML visual evidence: green boxes are ground truth, red boxes are detector outputs.

## What The Current Evidence Means

The current smoke reports prove that the software path is runnable:

- RAW simulation path is connected.
- HumanISP and PerceptionISP outputs are generated from the same RAW.
- A detector can be run on both outputs.
- Metrics and overlay images are produced.

They do not prove that PerceptionISP outperforms HumanISP. The current
single-image pseudo-label and small synthetic smoke runs are too narrow. The
auxiliary-map detector is a deterministic baseline rather than a trained
perception model, and the RGB+aux fusion adapter is only a reference integration
path until a detector is trained to consume those auxiliary channels.

## Aux Map DNN Training Status

Auxiliary maps are not automatically used by existing RGB DNNs. A pretrained
YOLO-style RGB detector expects three channels, so PerceptionISP aux maps become
useful only after one of these downstream changes:

- Train or fine-tune a detector with a six-channel RGB+aux input stem.
- Add a separate aux branch and fuse features with the RGB branch.
- Train a score/proposal calibration head that consumes aux evidence.

The repository now includes a DNN-facing export path:

- `perception_isp.aux_export`: writes `manifest.jsonl`, `labels/*.json`, and
  `tensors/*.npz`.
- `perception_isp.aux_train_smoke`: runs a tiny PyTorch optimization loop on
  the exported six-channel tensors, with optional deterministic train/eval
  split reporting.
- `RGBAuxTorchSmokeDetector`: loads the tiny checkpoint and feeds
  `perception_rgb_aux_dnn` metrics into the normal comparison harness.

Current local resource check:

- CUDA: unavailable.
- Apple MPS: available.
- RAM: 32 GB.

This is enough for smoke training and small subset fine-tuning, but not enough
to make strong perception performance claims from large-scale detector
retraining. The current verified smoke run exported two COCO8 CameraE2E-backed
samples in about 6 seconds.

A tiny RGB+aux stem benchmark on the exported COCO8 smoke manifest with
`--eval-fraction 0.5` ran on MPS at about 9.4 train sample-epochs/sec. The
same run recorded train and eval loss (`1.021 -> 1.005` train, `0.906 -> 0.903`
eval):

| Samples | Epochs | Estimated time |
| ---: | ---: | ---: |
| 10 | 3 | 3.2 s |
| 100 | 3 | 32.0 s |
| 1,000 | 3 | 5.3 min |
| 10,000 | 3 | 53.4 min |

This benchmark is a lower-bound smoke path for a tiny stem, not a full YOLO
fine-tuning estimate. Full detector training with a backbone, feature pyramid,
loss matching, and validation will be much slower and should use a CUDA GPU for
claim-quality experiments. The train-smoke path can record eval loss with
`--eval-fraction`, but the current result still proves only the data path and
basic optimization, not detector superiority.

The trained smoke checkpoint has also been run through the normal comparison
harness on two COCO8 CameraE2E-backed samples. The harness now reports a
`perception_rgb_aux_dnn` input, but its recall@0.50 is currently 0.000. That is
expected: this tiny checkpoint predicts one generic box and has no class head.
It proves that a learned RGB+aux path can be loaded and evaluated, not that it
is a useful detector yet.

### KITTI RGB+Aux Compact Dense Benchmark

A KITTI val subset was exported as six-channel RGB+aux tensors using
`detector_log`, `denoise=0.30`, `edge_aware`, `artifact=0.20`, and the cached
CameraE2E RAW samples:

```text
exports/perception_rgb_aux_kitti_val128_detector_log/index.html
```

Export result:

| Samples | Boxes | Export time | Export rate | RAW provenance |
| ---: | ---: | ---: | ---: | --- |
| 128 | 635 | 42.6 s | 3.00 samples/s | 128 true CFA mosaics, source/target `GRBG` |

The compact RGB+aux dense detector then trained on MPS:

| Run | Train/Eval | Epochs | Grid | Background weight | Elapsed | Sample-epochs/s | Best eval loss |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `dense` | 96 / 32 | 5 | 12x40 | 0.10 | 10.9 s | 44.0 | 1.0307 |
| `dense_bg1` | 96 / 32 | 8 | 12x40 | 1.00 | 13.7 s | 56.0 | 1.6490 |
| `rgb_aux_ablation` | 96 / 32 | 5 | 12x40 | 0.15 | 7.9 s | 60.7 | 1.0860 |
| `rgb_only_ablation` | 96 / 32 | 5 | 12x40 | 0.15 | 8.2 s | 58.6 | 1.1785 |
| `aux_only_ablation` | 96 / 32 | 5 | 12x40 | 0.15 | 8.1 s | 59.3 | 1.2003 |

The timing answer is therefore favorable for this compact path: on this Mac
with MPS, a 1,496-sample KITTI-val-sized run is estimated at about 2.8 minutes
for 5 epochs, and the 5,985-sample KITTI train split at about 11.3 minutes for
5 epochs. Tensor export remains a separate cost, roughly 0.33 s/sample for the
128-sample cached run.

However, the direct detector result is not useful yet:

| Run | Eval samples | Confidence | Precision@0.50 | Recall@0.50 | Small Recall@0.50 | FP@0.50 | Detections/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 32 | 0.30 | 0.0084 | 0.1558 | 0.1380 | 90.6563 | 91.4375 |
| `dense_bg1` | 32 | 0.30 | 0.0081 | 0.0990 | 0.0641 | 64.3750 | 64.8125 |
| `rgb_aux_ablation` | 32 | 0.30 | 0.0081 | 0.1047 | 0.0167 | 68.6250 | 69.2188 |
| `rgb_only_ablation` | 32 | 0.30 | 0.0089 | 0.0984 | 0.0312 | 59.8438 | 60.4688 |
| `aux_only_ablation` | 32 | 0.30 | 0.0045 | 0.0767 | 0.0156 | 92.5938 | 93.0000 |
| `rgb_aux_ablation` | 32 | 0.50 | 0.0114 | 0.0898 | 0.0167 | 48.7500 | 49.2812 |
| `rgb_only_ablation` | 32 | 0.50 | 0.0118 | 0.0835 | 0.0312 | 40.2500 | 40.8125 |
| `aux_only_ablation` | 32 | 0.50 | 0.0050 | 0.0767 | 0.0156 | 84.6250 | 85.0312 |

Confidence sweeps from `0.30` to `0.98` did not produce a usable operating
point. Raising confidence lowers FP but collapses recall. This is a concrete
negative result: the compact dense detector is useful to measure the RGB+aux
training path and resource needs, but it is not enough for a HumanISP-vs-
PerceptionISP superiority claim. A practical aux path should either fine-tune a
real detector stem/head or train a detector-side calibration branch over the
pretrained RGB detector proposals.

The channel ablation sharpens the conclusion. The `rgb_aux` model gets the best
eval loss and slightly higher R50 than `rgb_only`, but it does not improve the
operating point because FP and precision remain worse. `aux_only` is clearly
weaker. Therefore the current compact dense architecture is not a usable proof
that auxiliary maps improve detector performance; it is only a fast engineering
path for testing tensor export, training, checkpoint loading, and ablations.

## CameraE2E Resolution Finding

CameraE2E itself follows the expected path: RGB scene to spectral scene, sensor
CFA response to RAW, then ISP demosaic. The important issue is the bridge and
evaluation resolution:

- The bridge must consume `sensor.data["volts"]` as the true CFA mosaic, not as
  an RGB image or generic luma plane.
- The ISP should normally consume the CameraE2E sensor-native CFA pattern. The
  default evaluation setting is now `--cfa auto`, which keeps CameraE2E's GRBG
  source pattern as the PerceptionISP target pattern instead of remapping it to
  RGGB.
- The simulated scene/sensor plane must be large enough. A low native sensor
  plane upsampled later is not valid evidence for perception performance.
- The bridge now sets CameraE2E pixel pitch for evaluation so a 640x480 scene
  yields a native 640x480-ish sensor mosaic, then passes that true mosaic into
  PerceptionISP.

Each `RawFrame` now carries `raw_provenance`, and the evaluation reports expose:

- `raw_source_key`: for example `sensor.volts`.
- `source_shape`: native CameraE2E sensor mosaic shape.
- `requested_cfa_pattern`, `source_cfa_pattern`, and `target_cfa_pattern`.
- `pattern_remapped`.
- `true_sensor_cfa_mosaic`.
- `native_resolution_matches_target`.
- `native_resolution_at_least_target`.

The Bayer demosaic block now defaults to an edge-aware reference method. The
earlier bilinear/linear interpolation path remains available with
`--demosaic-method bilinear`.

COCO8 validation smoke results with CameraE2E sensor-native CFA
(`requested=auto`, source GRBG, target GRBG, remapped 0/4) and edge-aware
demosaic:

| Run | Reference RGB recall@0.50 | HumanISP recall@0.50 | Perception RGB recall@0.50 | RGB+Aux fusion recall@0.50 | Aux-only recall@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640x480 CameraE2E RAW | 0.628 | 0.500 | 0.500 | 0.500 | 0.000 |
| 1280x960 CameraE2E RAW | 0.678 | 0.578 | 0.578 | 0.578 | 0.000 |

An explicit RGGB remap smoke run produced the same recall numbers on this
COCO8 subset, but it reports `pattern_remapped_count=4/4`. The native-CFA run
is the preferred evidence path because it avoids an unnecessary bridge
transformation.

This supports the conclusion that scene/sensor resolution is a first-order
evaluation variable. The 1280x960 result is still not a superiority claim, but
it is a much more credible smoke test than the earlier low-resolution path.
The current fusion result matching Perception RGB means the aux evidence is
connected but has not yet improved this COCO8 smoke metric. The aux-only score
is 0.000 because the current aux baseline produces generic `object` proposals,
which do not match COCO labels in label-aware evaluation.

The current resolution sweep report is:

```text
reports/perception_resolution_sweep_coco8/index.html
```

## COCO YOLO Evidence

The strongest current evidence path uses real COCO images, CameraE2E-backed RAW
generation, the same YOLO11n detector on each RGB output, label-aware COCO
metrics, and no visualization overhead for larger runs. The setup is:

- `--source yolo-dataset`
- `--cfa auto`
- `--width 640 --height 480`
- `--rgb-detector yolo --rgb-detector-model yolo11n.pt`
- `--rgb-detector-confidence 0.25`
- `--label-aware`
- `--demosaic-method edge_aware`

Current aggregate results:

| Dataset | Samples | Input | Precision@0.50 | Recall@0.50 | Recall@0.75 | Small Recall@0.50 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| COCO128 train2017 | 128 | Reference RGB | 0.7059 | 0.5240 | 0.4616 | 0.0802 |
| COCO128 train2017 | 128 | HumanISP RGB | 0.6662 | 0.4697 | 0.3927 | 0.0519 |
| COCO128 train2017 | 128 | PerceptionISP RGB | 0.6728 | 0.4973 | 0.4373 | 0.0674 |
| COCO128 train2017 | 128 | RGB+Aux Fusion | 0.6749 | 0.4966 | 0.4365 | 0.0674 |
| COCO val2017 subset | 1,000 | Reference RGB | 0.6956 | 0.5391 | 0.4759 | 0.0697 |
| COCO val2017 subset | 1,000 | HumanISP RGB | 0.6621 | 0.4724 | 0.4035 | 0.0448 |
| COCO val2017 subset | 1,000 | PerceptionISP RGB | 0.6834 | 0.5034 | 0.4325 | 0.0544 |
| COCO val2017 subset | 1,000 | RGB+Aux Fusion | 0.6853 | 0.4992 | 0.4269 | 0.0542 |

On the 1,000-image COCO val subset, PerceptionISP RGB improves over HumanISP
RGB by:

- Precision@0.50: `+0.0213`
- Recall@0.50: `+0.0310`
- Small Recall@0.50: `+0.0095`

The current RGB+aux fusion is conservative: it keeps RGB detector labels and
uses aux evidence for score/filtering only. On the 1,000-image run it reduces
FP@0.50 slightly (`-0.034/sample` versus Perception RGB) and improves precision
slightly (`+0.0020`), but it loses recall (`-0.0043`). That is not enough to
claim that aux fusion is improving detector performance yet.

The important limitation is that COCO is a general object dataset, not a
driving-focused perception benchmark. These results are useful evidence that
the PerceptionISP RGB path can help a pretrained detector relative to the
HumanISP RGB rendering, but they are not sufficient for an automotive safety or
VRU claim. A driving dataset such as KITTI, BDD100K, nuImages, or a curated
CameraE2E driving-scene set is still required.

Current reports:

```text
reports/perception_yolo_coco_val2017_1k_camerae2e_fusion/index.html
reports/perception_yolo_coco_scale_rollup/index.html
```

The COCO val subset can be prepared reproducibly with:

```bash
PYTHONPATH=src python3 -m perception_isp.prepare_coco_subset \
  --output-dir data/coco_val2017_1k \
  --count 1000 \
  --split val2017 \
  --threads 16
```

The 1,000-image CameraE2E + YOLO run is long enough that larger runs should be
sharded with `--offset`, monitored with `--progress-interval`, and merged with
`perception_isp.merge_comparison_reports`. A single-shard merge smoke has been
verified against the 1k report and reproduced the original aggregate metrics
exactly.

## KITTI Driving Evidence

KITTI is a more relevant driving dataset than COCO, but the pretrained YOLO11n
detector is trained on COCO labels. For label-aware metrics, the evaluation must
map KITTI labels to compatible COCO labels:

```text
car -> car
van -> car
truck -> truck
pedestrian/person_sitting/Person_sitting -> person
cyclist -> bicycle
tram -> train
```

This is available as:

```bash
--ground-truth-label-map kitti-coco
```

The current KITTI val result uses the Ultralytics KITTI YOLO-format dataset,
CameraE2E-backed RAW, `640x192`, YOLO11n, `--label-aware`, and the
`kitti-coco` label map:

| Dataset | Samples | Input | Precision@0.50 | Recall@0.50 | Recall@0.75 | Small Recall@0.50 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| KITTI val | 1,496 | Reference RGB | 0.6189 | 0.5351 | 0.3616 | 0.3291 |
| KITTI val | 1,496 | HumanISP RGB | 0.6073 | 0.4695 | 0.3038 | 0.2794 |
| KITTI val | 1,496 | PerceptionISP RGB | 0.5991 | 0.4491 | 0.2890 | 0.2624 |
| KITTI val | 1,496 | RGB+Aux Fusion | 0.5974 | 0.4427 | 0.2851 | 0.2603 |

On KITTI val, PerceptionISP RGB is worse than HumanISP RGB:

- Precision@0.50: `-0.0082`
- Recall@0.50: `-0.0204`
- Small Recall@0.50: `-0.0170`

This is an important negative result. The COCO val subset suggests the
PerceptionISP RGB rendering can help a COCO detector, but the KITTI driving
result does not support a broad performance claim. The likely next work is to
tune the task-specific image formation for driving geometry and aspect ratio,
then rerun KITTI. The current aux fusion also does not help KITTI; it slightly
reduces detections and false positives but loses recall.

Current KITTI reports:

```text
reports/perception_yolo_kitti_val_1496_camerae2e_fusion/index.html
reports/perception_yolo_coco_kitti_rollup/index.html
```

The detector-facing tone fix was rerun on the full KITTI val set:

```text
reports/perception_isp_sweep_kitti_val_1496_detector_log_denoise030/index.html
reports/perception_yolo_kitti_detector_log_rollup/index.html
```

Full KITTI val with `tone=detector_log`, `denoise=0.30`, `edge_aware`,
`artifact=0.20`:

| Dataset | Samples | Input | Precision@0.50 | Recall@0.50 | Recall@0.75 | Small Recall@0.50 | FP@0.50 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| KITTI val | 1,496 | HumanISP RGB | 0.6073 | 0.4695 | 0.3038 | 0.2794 | 1.3409 |
| KITTI val | 1,496 | PerceptionISP RGB, default raw `log` | 0.5991 | 0.4491 | 0.2890 | 0.2624 | 1.2400 |
| KITTI val | 1,496 | PerceptionISP RGB, `detector_log` | 0.6022 | 0.4688 | 0.3057 | 0.2808 | 1.3750 |
| KITTI val | 1,496 | RGB+Aux Fusion, `detector_log` | 0.6063 | 0.4639 | 0.3030 | 0.2795 | 1.3235 |

Detector-log Perception RGB delta vs HumanISP:

- Precision@0.50: `-0.0051`
- Recall@0.50: `-0.0007`
- Recall@0.75: `+0.0019`
- Small Recall@0.50: `+0.0013`
- FP@0.50: `+0.0341`

Detector-log Perception RGB delta vs the previous default Perception RGB:

- Precision@0.50: `+0.0031`
- Recall@0.50: `+0.0197`
- Recall@0.75: `+0.0167`
- Small Recall@0.50: `+0.0184`
- FP@0.50: `+0.1350`

This changes the KITTI interpretation. The earlier negative result was largely
a detector-facing tone mismatch: raw `log` made YOLO lose recall. With
`detector_log`, PerceptionISP reaches near HumanISP recall parity and slightly
improves high-IoU/small-object recall, but it still trades away precision and
adds false positives. It is progress, not a broad superiority claim.

## KITTI Detector Score Threshold Sweep

The detector-log KITTI run was also analyzed with a post-hoc score threshold
sweep. This does not rerun CameraE2E or YOLO; it re-filters the saved detections
from the full 1,496-sample report:

```text
reports/perception_threshold_sweep_kitti_val_1496_detector_log/index.html
```

The sweep shows that score calibration alone does not solve the tradeoff. At
the default saved threshold, PerceptionISP RGB is within `-0.0007` R50 of
HumanISP and improves R75/small recall slightly, but FP is higher. Raising the
threshold reduces FP, but recall and small-object recall drop too quickly.

| Input | Score threshold | Precision@0.50 | Recall@0.50 | Delta R50 | Small Recall@0.50 | Delta Small | FP@0.50 | Delta FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PerceptionISP RGB | 0.250 | 0.6022 | 0.4688 | -0.0007 | 0.2808 | +0.0013 | 1.3750 | +0.0341 |
| PerceptionISP RGB | 0.275 | 0.6202 | 0.4529 | -0.0166 | 0.2622 | -0.0173 | 1.1464 | -0.1945 |
| PerceptionISP RGB | 0.300 | 0.6285 | 0.4347 | -0.0348 | 0.2403 | -0.0391 | 0.9753 | -0.3656 |
| RGB+Aux Fusion | 0.250 | 0.6109 | 0.4573 | -0.0122 | 0.2726 | -0.0069 | 1.2413 | -0.0996 |
| RGB+Aux Fusion | 0.275 | 0.6218 | 0.4411 | -0.0283 | 0.2526 | -0.0268 | 1.0648 | -0.2761 |
| RGB+Aux Fusion | 0.300 | 0.6286 | 0.4255 | -0.0440 | 0.2362 | -0.0433 | 0.8997 | -0.4412 |

With the recall floor set to `delta R50 >= -0.001`, the best option remains
PerceptionISP RGB at `0.250`; no higher threshold satisfies the floor. If FP
must be lower than HumanISP, the best-recall option is RGB+Aux Fusion at
`0.250`, but it costs `-0.0122` R50. This makes a trained detector-side aux
adapter more plausible than simple threshold tuning, but it also means any
claim needs a trained and held-out validation result.

## KITTI Proposal Calibration

A lightweight detector-side proposal calibrator was added to test the next
lowest-cost aux integration path. It trains logistic score calibration on saved
`perception_fusion_rgb_aux` proposals and evaluates on a held-out sample split,
without rerunning CameraE2E or YOLO:

```text
reports/perception_proposal_calibration_kitti_val_1496_detector_log/index.html
reports/perception_proposal_calibration_kitti_val_1496_detector_log/proposal_calibration_model.json
reports/perception_calibrated_fusion_kitti_val_1496_detector_log_eval/index.html
reports/perception_calibrated_fusion_kitti_val_1496_detector_log/index.html
```

Setup:

| Item | Value |
| --- | --- |
| Source report | Full KITTI val detector-log run |
| Input proposals | `perception_fusion_rgb_aux` |
| Split | Hash split, 1,039 train / 457 eval samples |
| Train proposals | 2,534 positive / 1,394 negative |
| Eval proposals | 1,115 positive / 586 negative |
| Feature sets | `score_aux`, `score_label`, `score_label_aux` |

Held-out eval baseline:

| Eval input | Precision@0.50 | Recall@0.50 | Recall@0.75 | Small Recall@0.50 | FP@0.50 | Detections/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HumanISP RGB | 0.6084 | 0.4573 | 0.2850 | 0.2641 | 1.2867 | 3.7593 |
| Original RGB+Aux Fusion | 0.6027 | 0.4495 | 0.2802 | 0.2672 | 1.2823 | 3.7221 |

Best low-recall-loss calibration point against the original fusion input:

| Feature set | Threshold | Precision@0.50 | Recall@0.50 | Recall@0.75 | Small Recall@0.50 | FP@0.50 | Delta P vs Original | Delta R vs Original | Delta FP vs Original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `score_label_aux` | 0.02 | 0.6334 | 0.4492 | 0.2802 | 0.2672 | 1.0306 | +0.0307 | -0.0002 | -0.2516 |

The model artifact can now be applied back to a comparison report as
`perception_calibrated_fusion_rgb_aux`. The held-out apply report uses only the
457 eval samples from the calibration split and reproduces the calibration
metrics exactly. The full 1,496-sample apply report is useful to inspect the
operating point across all available samples, but it includes training samples
and should not be treated as held-out evidence.

The normal comparison, ISP sweep, and resolution sweep CLIs now also accept
`--proposal-calibration-model`, which applies the same artifact during the run
instead of requiring a separate saved-report apply step.

Live normal-harness verification:

```text
reports/perception_compare_kitti_val_1496_detector_log_calibrated_harness/index.html
```

This run uses the same KITTI val 1,496 samples, `detector_log` PerceptionISP
config, fixed `log` HumanISP baseline, label-aware KITTI-to-COCO labels, true
CameraE2E RAW cache, and the saved calibration artifact. The aggregate metrics
match the full post-process apply report exactly for `human_rgb`,
`perception_rgb`, `perception_fusion_rgb_aux`, and
`perception_calibrated_fusion_rgb_aux`.

Applied report results:

| Report split | Input | Precision@0.50 | Recall@0.50 | Recall@0.75 | Small Recall@0.50 | FP@0.50 | Detections/sample |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eval 457 | HumanISP RGB | 0.6084 | 0.4573 | 0.2850 | 0.2641 | 1.2867 | 3.7593 |
| Eval 457 | RGB+Aux Fusion | 0.6027 | 0.4495 | 0.2802 | 0.2672 | 1.2823 | 3.7221 |
| Eval 457 | Calibrated RGB+Aux Fusion | 0.6334 | 0.4492 | 0.2802 | 0.2672 | 1.0306 | 3.4683 |
| All 1,496 | HumanISP RGB | 0.6073 | 0.4695 | 0.3038 | 0.2794 | 1.3409 | 3.8135 |
| All 1,496 | RGB+Aux Fusion | 0.6063 | 0.4639 | 0.3030 | 0.2795 | 1.3235 | 3.7627 |
| All 1,496 | Calibrated RGB+Aux Fusion | 0.6370 | 0.4633 | 0.3027 | 0.2795 | 1.0729 | 3.5094 |

This is useful progress, but it is not a HumanISP superiority result. On the
same eval split, the calibrated result is still `-0.0081` R50 versus HumanISP.
It does improve precision and FP versus HumanISP (`+0.0250` P50, `-0.2560`
FP50), while keeping small-object recall slightly above HumanISP (`+0.0031`).

The ablation also matters:

| Feature set | Threshold | Precision@0.50 | Recall@0.50 | FP@0.50 | Delta P vs Original | Delta R vs Original | Delta FP vs Original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `score_aux` | 0.02 | 0.6029 | 0.4495 | 1.2757 | +0.0002 | +0.0000 | -0.0066 |
| `score_label` | 0.02 | 0.6331 | 0.4492 | 1.0328 | +0.0304 | -0.0002 | -0.2495 |
| `score_label_aux` | 0.02 | 0.6334 | 0.4492 | 1.0306 | +0.0307 | -0.0002 | -0.2516 |

The main win is label/proposal calibration, with aux evidence adding only a
small extra improvement on this run. That is the right caveat: aux maps are
connected to a useful calibration path, but the current evidence does not show
that aux alone is driving the gain.

## KITTI ISP Tuning Sweep

The first KITTI tuning pass keeps the HumanISP baseline fixed at the default
evaluation config, then sweeps only PerceptionISP image formation settings. This
avoids a misleading comparison where both HumanISP and PerceptionISP move when
the same runtime config changes.

KITTI val 64-sample sweep:

```text
reports/perception_isp_sweep_kitti_val_64/index.html
```

Sweep grid:

```text
tone_mapping: log, srgb, linear
denoise_strength: 0.0, 0.18, 0.30
demosaic_method: edge_aware
demosaic_artifact_suppression: 0.20
```

Best Perception RGB candidate on this small subset:

| Candidate | Human Recall@0.50 | Perception Recall@0.50 | Delta Recall@0.50 | Delta Precision@0.50 | Delta Recall@0.75 | Delta Small Recall@0.50 | Delta FP@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tone=srgb`, `denoise=0.30`, `edge_aware`, `artifact=0.20` | 0.4749 | 0.4865 | +0.0116 | +0.0270 | -0.0064 | +0.0221 | -0.1563 |

This is a useful tuning signal, not a performance claim. The same candidate
did not hold up at 512 samples. The follow-up 512-sample sweep used the same
fixed HumanISP baseline and CameraE2E RAW cache:

```text
reports/perception_isp_sweep_kitti_val_512_log_srgb_denoise/index.html
```

512-sample sweep grid:

```text
tone_mapping: log, srgb
denoise_strength: 0.0, 0.18, 0.30
demosaic_method: edge_aware
demosaic_artifact_suppression: 0.20
```

Best Perception RGB recall candidate at 512 samples:

| Candidate | Human Recall@0.50 | Perception Recall@0.50 | Delta Recall@0.50 | Delta Precision@0.50 | Delta Recall@0.75 | Delta Small Recall@0.50 | Delta FP@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tone=srgb`, `denoise=0.00`, `edge_aware`, `artifact=0.20` | 0.4803 | 0.4740 | -0.0062 | +0.0068 | -0.0042 | -0.0108 | -0.0645 |

So far, KITTI tuning improves precision and reduces false positives mainly by
reducing detections, but recall remains below HumanISP. This is the result that
should drive the next work: broad performance claims are not justified, and the
next step should be either a deeper image-formation change or task-specific
DNN fine-tuning, not just small scalar tuning.

A follow-up detector-facing tone test found the likely reason for a large part
of the KITTI recall loss: PerceptionISP `log` was a raw log tone curve, while
HumanISP `log` was gamma-encoded for display/detector input. The gamma-encoded
log curve is now exposed as `detector_log` for PerceptionISP runs. The 512-sample
single-candidate report is:

```text
reports/perception_isp_sweep_kitti_val_512_detector_log_denoise030/index.html
```

Best detector-safe tone candidate at 512 samples:

| Candidate | Human Recall@0.50 | Perception Recall@0.50 | Delta Recall@0.50 | Delta Precision@0.50 | Delta Recall@0.75 | Delta Small Recall@0.50 | Delta FP@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tone=detector_log`, `denoise=0.30`, `edge_aware`, `artifact=0.20` | 0.4803 | 0.4803 | +0.0001 | -0.0047 | +0.0037 | +0.0005 | +0.0254 |

This recovers recall parity and slightly improves Recall@0.75/small Recall,
but it increases false positives and lowers precision. It is a better direction
than `srgb`, but still not a clean win.

The 512 run also exposed and then verified an experiment tooling fix. The first
512 CameraE2E RAW cache fill took `474.8s` at `1.08 samples/s`; reloading the
same 512 cached samples took `8.9s` at `57.8 samples/s`. The loaders and CLIs
now have `--load-progress-interval` to make the raw-preparation phase visible
and `--raw-cache-dir` to reuse already generated dataset RAW samples across
tuning and validation reruns.

## Required Before Performance Claims

Use a real labeled driving dataset subset and report at least:

- Overall recall and precision.
- Small-object recall.
- VRU recall for pedestrians and cyclists.
- Traffic-light recall or state accuracy if the labels support it.
- Runtime and output tensor bandwidth.

For a fair PerceptionISP claim, compare at least:

- HumanISP RGB + pretrained RGB detector.
- PerceptionISP RGB + the same pretrained RGB detector.
- PerceptionISP RGB+aux + a detector or adapter trained to consume those maps.

The current `RGBAuxTorchSmokeDetector` is not sufficient for that final bullet;
it is only the first learned-adapter plumbing step.

## Recommended Next Run

Use a KITTI object-detection subset:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source kitti-dataset \
  --dataset /path/to/KITTI/object \
  --split training \
  --count 32 \
  --width 640 --height 192 \
  --rgb-detector yolo \
  --output-dir reports/perception_compare_kitti
```

Use `--no-camerae2e` only to debug dataset parsing quickly. Leave CameraE2E
enabled for evidence that is meant to represent the simulation path.
