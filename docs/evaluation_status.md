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
