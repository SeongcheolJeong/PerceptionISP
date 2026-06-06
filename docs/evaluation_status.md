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
