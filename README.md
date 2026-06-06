# PerceptionISP

Software reference implementation of a perception-oriented automotive ISP.

This repo follows the design notes in `camera_sim_perception_isp_full_conversation_summary.md` and the attached Perception ISP text:

- RGB-compatible vision stream is kept for existing DNN backbones.
- Sensor-native auxiliary maps are preserved: noise, SNR, saturation, HDR source, edge, color confidence, IR/Clear, blur/focus, flicker, timing.
- Fast safety path and accurate autonomy path are separate outputs.
- CameraE2E under `/Users/seongcheoljeong/Documents/CameraE2E` can be used opportunistically, with synthetic RAW fallback.

## Quick Run

```bash
PYTHONPATH=src python3 -m perception_isp.cli --output-dir reports/perception_isp_demo
```

Try other CFA modes:

```bash
PYTHONPATH=src python3 -m perception_isp.cli --cfa RCCB
PYTHONPATH=src python3 -m perception_isp.cli --cfa RGBIR
PYTHONPATH=src python3 -m perception_isp.cli --cfa MONO
```

Try CameraE2E first, then synthetic fallback:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
python3 -m perception_isp.cli --camerae2e --scene "uniform ee"
```

## Implemented Blocks

1. Sensor Interface / Metadata Capture
2. Calibration Loader
3. RAW Physical Normalization
4. Noise / Uncertainty Engine
5. CFA / Pixel Structure Decoder
6. HDR / Exposure Fusion Engine
7. Edge / Structure Engine
8. Color / Spectral Engine
9. Optics / Geometry / Timing Engine
10. LED Flicker / Temporal Engine
11. Task-specific Image Formation
12. Output Formatter
13. Runtime Controller
14. Safety / health monitor

## Outputs

The CLI writes:

- `perception_isp_outputs.npz`: vision RGB, accurate tensor, fast tensor, maps.
- `summary.json`: channel names, metadata, health, latency estimate.
- `vision_rgb.ppm` and `human_rgb.ppm`: simple preview images.

The accurate tensor uses channels:

```text
rgb_r, rgb_g, rgb_b,
noise_variance, saturation, edge_strength, edge_confidence,
hdr_exposure_source, ir_or_clear, blur_focus_confidence
```

The fast tensor uses channels:

```text
luma, edge_strength, edge_confidence,
temporal_difference, saturation, noise_variance
```

## RGB+Aux DNN Export

Existing RGB detectors do not automatically use PerceptionISP auxiliary maps.
To make aux maps useful, the downstream model must be adapted and trained or
fine-tuned with aux channels. The export path writes a six-channel tensor:

```text
rgb_r, rgb_g, rgb_b,
aux_edge_strength, aux_saturation, aux_reliability
```

Export a small CameraE2E-backed COCO8 smoke dataset:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.aux_export \
  --source yolo-dataset \
  --dataset data/coco8/data.yaml \
  --split val \
  --count 2 \
  --width 640 --height 480 \
  --cfa auto \
  --tone-mapping srgb \
  --demosaic-method edge_aware \
  --demosaic-artifact-suppression 0.35 \
  --output-dir exports/perception_rgb_aux_coco8_smoke
```

Run a tiny PyTorch smoke training loop to prove the six-channel tensor can be
consumed by a DNN stem and optimized:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.aux_train_smoke \
  --manifest exports/perception_rgb_aux_coco8_smoke/manifest.jsonl \
  --epochs 3 \
  --device auto \
  --eval-fraction 0.5 \
  --estimate-samples 10,100,1000,10000 \
  --output-dir exports/perception_rgb_aux_coco8_train_benchmark
```

This smoke training is not a detector-performance claim. It only verifies the
data path needed before real RGB+aux detector training. The summary includes
elapsed time, sample-epochs/sec, and simple time estimates for the requested
sample counts. With `--eval-fraction`, it also records a deterministic
train/eval split and eval loss. These estimates apply to the tiny RGB+aux stem
only; full detector fine-tuning will be much slower.

Run the trained smoke checkpoint through the normal comparison harness:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source yolo-dataset \
  --dataset data/coco8/data.yaml \
  --split val \
  --count 2 \
  --width 640 --height 480 \
  --cfa auto \
  --rgb-detector yolo \
  --tone-mapping srgb \
  --demosaic-method edge_aware \
  --rgb-aux-detector-checkpoint exports/perception_rgb_aux_coco8_train_benchmark/rgb_aux_smoke_detector.pt \
  --output-dir reports/perception_compare_coco8_rgb_aux_dnn_smoke
```

The resulting `perception_rgb_aux_dnn` metric is expected to be weak until a
real detector head is trained. The smoke checkpoint predicts one generic box
and does not learn class labels.

## Validation

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## HumanISP vs PerceptionISP Evaluation Harness

Run a lightweight synthetic A/B comparison:

```bash
PYTHONPATH=src python3 -m perception_isp.eval_cli \
  --source synthetic \
  --output-dir reports/perception_compare_synthetic
```

Run the labeled synthetic scene through CameraE2E first:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source camerae2e-synthetic \
  --output-dir reports/perception_compare_camerae2e
```

The default detector is a pure-numpy smoke detector. Use `--rgb-detector yolo`
after installing `ultralytics` and `torch` to run a real pretrained detector.
The HTML report writes `assets/*.png` overlays by default: green boxes are
ground truth and red boxes are detector outputs. Add `--no-visuals` only when
you need a metric-only batch run.

Run a real-image detector-consistency smoke test using YOLO pseudo labels:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source sample-image \
  --rgb-detector yolo \
  --width 320 --height 240 \
  --output-dir reports/perception_compare_sample_image
```

This uses detector-generated pseudo labels, not human ground truth. It proves
the real-image + CameraE2E + ISP + detector path, but it is not sufficient for
performance claims.

Run a YOLO-format labeled dataset, such as a KITTI or BDD subset converted to
Ultralytics layout:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source yolo-dataset \
  --dataset /path/to/data.yaml \
  --split val \
  --count 16 \
  --rgb-detector yolo \
  --tone-mapping srgb \
  --output-dir reports/perception_compare_yolo_dataset
```

Use `--no-camerae2e` only for fast adapter tests. CameraE2E-backed evidence
should leave it off.

Prepare a reproducible COCO val2017 subset without downloading full COCO train:

```bash
PYTHONPATH=src python3 -m perception_isp.prepare_coco_subset \
  --output-dir data/coco_val2017_1k \
  --count 1000 \
  --split val2017 \
  --threads 16
```

Run the 1k CameraE2E + YOLO comparison:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source yolo-dataset \
  --dataset data/coco_val2017_1k/data.yaml \
  --split val \
  --offset 0 \
  --count 1000 \
  --width 640 --height 480 \
  --cfa auto \
  --rgb-detector yolo \
  --rgb-detector-model yolo11n.pt \
  --rgb-detector-confidence 0.25 \
  --label-aware \
  --no-visuals \
  --demosaic-method edge_aware \
  --output-dir reports/perception_yolo_coco_val2017_1k_camerae2e_fusion
```

Roll up multiple comparison reports:

```bash
PYTHONPATH=src python3 -m perception_isp.report_rollup \
  reports/perception_yolo_coco128_128_camerae2e_fusion \
  reports/perception_yolo_coco_val2017_1k_camerae2e_fusion \
  --output-dir reports/perception_yolo_coco_scale_rollup
```

For long runs, shard with `--offset`, print progress, then merge the shards:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source yolo-dataset \
  --dataset data/coco_val2017_1k/data.yaml \
  --split val \
  --offset 0 \
  --count 250 \
  --width 640 --height 480 \
  --cfa auto \
  --rgb-detector yolo \
  --label-aware \
  --no-visuals \
  --progress-interval 25 \
  --output-dir reports/perception_yolo_coco_val2017_1k_shard_0000

PYTHONPATH=src python3 -m perception_isp.merge_comparison_reports \
  reports/perception_yolo_coco_val2017_1k_shard_0000 \
  reports/perception_yolo_coco_val2017_1k_shard_0250 \
  reports/perception_yolo_coco_val2017_1k_shard_0500 \
  reports/perception_yolo_coco_val2017_1k_shard_0750 \
  --name coco_val_1k_sharded \
  --output-dir reports/perception_yolo_coco_val2017_1k_merged
```

Run a resolution sweep and summary report:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.resolution_sweep \
  --source yolo-dataset \
  --dataset data/coco8/data.yaml \
  --split val \
  --count 4 \
  --resolutions 640x480,1280x960 \
  --rgb-detector yolo \
  --label-aware \
  --tone-mapping srgb \
  --demosaic-method edge_aware \
  --demosaic-artifact-suppression 0.35 \
  --output-dir reports/perception_resolution_sweep_coco8
```

The sweep summary checks whether each sample used true CameraE2E sensor CFA
data, whether CameraE2E source CFA matches the ISP target CFA, and whether
native sensor resolution was at least the requested target. The default
`--cfa auto` keeps CameraE2E's sensor-native CFA pattern; use an explicit
pattern such as `--cfa RGGB` only when intentionally testing remap behavior.
It also includes `perception_fusion_rgb_aux`, a conservative RGB+aux adapter
that keeps the RGB detector class labels and uses PerceptionISP edge,
saturation, and reliability maps as support evidence. This is a reference
integration path, not a learned aux-map detector.

Bayer demosaic defaults to `edge_aware`. Use `--demosaic-method bilinear` to
reproduce the earlier simple linear interpolation baseline.

For CameraE2E-backed perception metrics, use a scene/sensor resolution that is
large enough for the detector task. COCO8 smoke runs showed that 640x480 can be
too low after RAW/CFA simulation for small or thin objects, while 1280x960
recovers substantially more detector recall:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source yolo-dataset \
  --dataset data/coco8/data.yaml \
  --split val \
  --count 4 \
  --width 1280 --height 960 \
  --rgb-detector yolo \
  --label-aware \
  --tone-mapping srgb \
  --demosaic-method edge_aware \
  --demosaic-artifact-suppression 0.35 \
  --output-dir reports/perception_compare_coco8_val_1280
```

Run a native KITTI object-detection subset without converting labels:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source kitti-dataset \
  --dataset /path/to/KITTI/object \
  --split training \
  --count 16 \
  --rgb-detector yolo \
  --output-dir reports/perception_compare_kitti
```

Supported KITTI layouts are `training/image_2` + `training/label_2` and compact
`image_2` + `label_2` subsets.

Run the Ultralytics YOLO-format KITTI dataset with COCO YOLO label remapping:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source yolo-dataset \
  --dataset data/kitti/data.yaml \
  --split val \
  --count 1496 \
  --width 640 --height 192 \
  --cfa auto \
  --rgb-detector yolo \
  --label-aware \
  --ground-truth-label-map kitti-coco \
  --no-visuals \
  --progress-interval 374 \
  --load-progress-interval 374 \
  --output-dir reports/perception_yolo_kitti_val_1496_camerae2e_fusion
```

The `kitti-coco` label map is required because YOLO11n emits COCO labels such
as `person` and `bicycle`, while KITTI labels use `pedestrian`, `cyclist`, and
`van`.

Before spending time on full KITTI reruns, sweep the PerceptionISP image
formation settings against a fixed HumanISP baseline:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.isp_sweep \
  --source yolo-dataset \
  --dataset data/kitti/data.yaml \
  --split val \
  --count 64 \
  --width 640 --height 192 \
  --cfa auto \
  --rgb-detector yolo \
  --label-aware \
  --ground-truth-label-map kitti-coco \
  --no-visuals \
  --load-progress-interval 16 \
  --tone-mappings log,srgb,linear \
  --denoise-strengths 0.0,0.18,0.30 \
  --demosaic-methods edge_aware \
  --demosaic-artifact-suppressions 0.20 \
  --output-dir reports/perception_isp_sweep_kitti_val_64
```

The sweep report keeps the HumanISP baseline fixed while PerceptionISP settings
change, then ranks configs by delta recall against HumanISP. Treat the best
subset result as a candidate, then rerun it on the full validation set.

This is a runnable SW reference, not a product ISP. The intentional next step is to compare these outputs against task metrics such as small-object recall, VRU recall, traffic-light state accuracy, and AEB early-warning lead time.
