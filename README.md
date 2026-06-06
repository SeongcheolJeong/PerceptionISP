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
noise_variance, snr_map, saturation, clipping_distance, hdr_confidence,
edge_strength, edge_confidence, demosaic_confidence,
hdr_exposure_source, lens_gain, color_confidence,
ir_or_clear, blur_focus_confidence
```

The fast tensor uses channels:

```text
luma, edge_strength, edge_confidence,
temporal_difference, saturation, noise_variance
```

## RGB+Aux DNN Export

Existing RGB detectors do not automatically use PerceptionISP auxiliary maps.
To make aux maps useful, the downstream model must be adapted and trained or
fine-tuned with aux channels. The export path writes a stable six-channel
tensor for the current compact detector path:

```text
rgb_r, rgb_g, rgb_b,
aux_edge_strength, aux_saturation, aux_reliability
```

It also stores an extended sensor-native tensor for new aux-aware experiments:

```text
rgb_r, rgb_g, rgb_b,
aux_edge_strength, aux_saturation, aux_reliability,
aux_noise_risk, aux_clipping_distance, aux_demosaic_confidence,
aux_hdr_confidence, aux_lens_gain, aux_color_confidence,
aux_blur_focus_confidence
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
  --load-progress-interval 1 \
  --raw-cache-dir data/.cache/perception_isp_raw \
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

Use `--tensor-key rgb_aux_extended_chw` to train the same smoke path on the
extended 13-channel sensor-native tensor. The default remains `rgb_aux_chw` so
existing checkpoints and timing reports stay reproducible.

This smoke training is not a detector-performance claim. It only verifies the
data path needed before real RGB+aux detector training. The summary includes
elapsed time, sample-epochs/sec, and simple time estimates for the requested
sample counts. With `--eval-fraction`, it also records a deterministic
train/eval split and eval loss. These estimates apply to the tiny RGB+aux stem
only; full detector fine-tuning will be much slower.

For a driving-sized local timing check, export and train a compact KITTI
RGB+aux detector on 128 cached CameraE2E-backed validation samples:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.aux_export \
  --source yolo-dataset \
  --dataset data/kitti/data.yaml \
  --split val \
  --count 128 \
  --width 640 --height 192 \
  --cfa auto \
  --load-progress-interval 32 \
  --raw-cache-dir data/.cache/perception_isp_raw \
  --tone-mapping detector_log \
  --denoise-strength 0.30 \
  --demosaic-method edge_aware \
  --demosaic-artifact-suppression 0.20 \
  --output-dir exports/perception_rgb_aux_kitti_val128_detector_log

PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.aux_train_dense \
  --manifest exports/perception_rgb_aux_kitti_val128_detector_log/manifest.jsonl \
  --epochs 5 \
  --device auto \
  --grid 12x40 \
  --base-channels 16 \
  --tensor-key rgb_aux_chw \
  --channel-mode rgb_aux \
  --eval-fraction 0.25 \
  --include-labels car,pedestrian,cyclist \
  --estimate-samples 128,1496,5985 \
  --output-dir exports/perception_rgb_aux_kitti_val128_detector_log_dense
```

This compact dense detector is still a learning-path benchmark, not a
claim-quality detector. The current KITTI 128 run trains quickly on MPS, but
the direct detector metrics are weak and produce too many false positives.
Use `--channel-mode rgb_only` or `--channel-mode aux_only` for ablations with
the same selected tensor shape and zeroed input groups. Use
`--tensor-key rgb_aux_extended_chw` to train the compact dense detector on the
extended 13-channel sensor-native tensor.

The current MPS timing is good for iteration, not for a performance claim. From
the observed KITTI ablation runs, compact dense training is about
`59 sample-epochs/s`: a 5,985-sample, 5-epoch compact run is about 8.4 minutes
for training only, 50 epochs is about 1.4 hours, and 100 epochs is about
2.8 hours. Tensor export is separate from training and is currently about
0.33 s/sample on the cached 128-sample KITTI run, so export plus 5-epoch
training is about 41.6 minutes for 5,985 images.

Roll up export, training, and dense-eval summaries into one timing/diagnostic
report:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.aux_training_rollup \
  exports/perception_rgb_aux_kitti_val128_detector_log \
  exports/perception_rgb_aux_kitti_val128_detector_log_dense_rgb_aux_ablation \
  exports/perception_rgb_aux_kitti_val128_detector_log_dense_rgb_only_ablation \
  exports/perception_rgb_aux_kitti_val128_detector_log_dense_aux_only_ablation \
  reports/perception_rgb_aux_dense_kitti_val128_rgb_aux_ablation_eval_conf050 \
  reports/perception_rgb_aux_dense_kitti_val128_rgb_only_ablation_eval_conf050 \
  reports/perception_rgb_aux_dense_kitti_val128_aux_only_ablation_eval_conf050 \
  --output-dir reports/perception_rgb_aux_training_rollup_kitti_val128
```

The rollup is a resource and diagnostic view. It now includes a Training-Time
Plan derived from observed sample-epochs/sec and export samples/sec, making
compact KITTI-sized timing scenarios reproducible. It also makes clear that the
compact dense path trains quickly, while its direct detector metrics are still
too weak for a HumanISP-vs-PerceptionISP performance claim.

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
  --raw-cache-dir data/.cache/perception_isp_raw \
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
  --raw-cache-dir data/.cache/perception_isp_raw \
  --tone-mappings log,detector_log,srgb,linear \
  --denoise-strengths 0.0,0.18,0.30 \
  --demosaic-methods edge_aware \
  --demosaic-artifact-suppressions 0.20 \
  --output-dir reports/perception_isp_sweep_kitti_val_64
```

The sweep report keeps the HumanISP baseline fixed while PerceptionISP settings
change, then ranks configs by delta recall against HumanISP. Treat the best
subset result as a candidate, then rerun it on the full validation set.
For pretrained RGB detectors, include `detector_log`; it is the gamma-encoded
log tone curve intended to preserve detector-facing contrast.

Run the current full KITTI detector-facing candidate:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.isp_sweep \
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
  --tone-mappings detector_log \
  --denoise-strengths 0.30 \
  --demosaic-methods edge_aware \
  --demosaic-artifact-suppressions 0.20 \
  --load-progress-interval 187 \
  --progress-interval 374 \
  --raw-cache-dir data/.cache/perception_isp_raw \
  --output-dir reports/perception_isp_sweep_kitti_val_1496_detector_log_denoise030
```

After a detector run finishes, sweep saved detector scores without rerunning
CameraE2E or YOLO:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.threshold_sweep \
  reports/perception_isp_sweep_kitti_val_1496_detector_log_denoise030/001_tone-detector-log_denoise-0.30_demosaic-edge-aware_artifact-0.20 \
  --inputs perception_rgb,perception_fusion_rgb_aux \
  --thresholds 0.25:0.55:0.025 \
  --baseline-input human_rgb \
  --recall-delta-floor -0.001 \
  --output-dir reports/perception_threshold_sweep_kitti_val_1496_detector_log
```

This only re-filters detections that already exist in the saved report. It is
useful for checking whether PerceptionISP's FP/precision tradeoff can be fixed
with detector score calibration before spending time on DNN training.

Train/evaluate a lightweight proposal calibrator on a saved detector report:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.proposal_calibration \
  reports/perception_isp_sweep_kitti_val_1496_detector_log_denoise030/001_tone-detector-log_denoise-0.30_demosaic-edge-aware_artifact-0.20 \
  --input perception_fusion_rgb_aux \
  --feature-sets score_aux,score_label,score_label_aux \
  --thresholds 0.00:0.30:0.01 \
  --baseline-input human_rgb \
  --train-fraction 0.70 \
  --split-strategy hash \
  --epochs 800 \
  --lr 0.05 \
  --l2 0.001 \
  --write-feature-artifacts \
  --output-dir reports/perception_proposal_calibration_kitti_val_1496_detector_log
```

This is a detector-side calibration branch over saved RGB proposals. It is a
cheap way to test whether aux evidence can help score/filtering before training
a full detector.

The calibration report also writes `proposal_calibration_model.json`.
`--write-feature-artifacts` additionally writes
`proposal_calibration_model_score_aux.json`,
`proposal_calibration_model_score_label.json`, and
`proposal_calibration_model_score_label_aux.json`, so feature-set ablations can
be applied back to the same comparison report with distinct output input names.
Use `--artifact-feature-set`, `--artifact-threshold`, and
`--artifact-output-input` when the primary `proposal_calibration_model.json`
needs to target a specific row. Apply an artifact back to a comparison report as
a new calibrated input:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.proposal_calibration_apply \
  reports/perception_isp_sweep_kitti_val_1496_detector_log_denoise030/001_tone-detector-log_denoise-0.30_demosaic-edge-aware_artifact-0.20 \
  --model reports/perception_proposal_calibration_kitti_val_1496_detector_log/proposal_calibration_model.json \
  --split eval \
  --output-dir reports/perception_calibrated_fusion_kitti_val_1496_detector_log_eval
```

Apply several feature-specific artifacts and generate a rollup in one pass:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.proposal_calibration_apply \
  reports/perception_compare_kitti_val_1496_detector_log_calibrated_harness \
  --model reports/perception_proposal_calibration_kitti_train512_detector_log/proposal_calibration_model_score_aux.json \
  --model reports/perception_proposal_calibration_kitti_train512_detector_log/proposal_calibration_model_score_label.json \
  --model reports/perception_proposal_calibration_kitti_train512_detector_log/proposal_calibration_model_score_label_aux.json \
  --split all \
  --output-dir reports/perception_calibrated_fusion_kitti_train512_to_val1496_features \
  --rollup-output-dir reports/perception_train512_calibration_feature_ablation_rollup \
  --rollup-baseline-input perception_fusion_rgb_aux \
  --include-source-report-in-rollup
```

The rollup names calibrated runs by proposal feature set and adds the report
directory when two calibrated runs would otherwise have the same display name.
Use `--rollup-baseline-input perception_fusion_rgb_aux` for feature ablations,
because the key question is improvement over the uncalibrated RGB+Aux fusion
input. The rollup includes baseline-relative precision, recall, small-recall,
false-positive, and detection-count deltas.

Use `--split eval` for held-out evidence from the original calibration split.
Use `--split all` only as an operational full-report application, because it
includes samples used to fit the calibrator.

The same model can also be applied during normal evaluation and sweep runs, so
the generated comparison report contains `perception_calibrated_fusion_rgb_aux`
without a separate post-process step:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.eval_cli \
  --source yolo-dataset \
  --dataset data/kitti/data.yaml \
  --split val \
  --count 1496 \
  --width 640 \
  --height 192 \
  --cfa auto \
  --rgb-detector yolo \
  --rgb-detector-model yolo11n.pt \
  --rgb-detector-confidence 0.25 \
  --label-aware \
  --ground-truth-label-map kitti-coco \
  --no-visuals \
  --tone-mapping detector_log \
  --denoise-strength 0.30 \
  --demosaic-method edge_aware \
  --demosaic-artifact-suppression 0.20 \
  --human-tone-mapping log \
  --human-denoise-strength 0.18 \
  --human-demosaic-method edge_aware \
  --human-demosaic-artifact-suppression 0.20 \
  --raw-cache-dir data/.cache/perception_isp_raw \
  --proposal-calibration-model reports/perception_proposal_calibration_kitti_val_1496_detector_log/proposal_calibration_model.json \
  --output-dir reports/perception_compare_kitti_val_1496_detector_log_calibrated_harness
```

`--proposal-calibration-model` requires fusion to stay enabled because the
artifact is trained on `perception_fusion_rgb_aux` proposals.

For stricter evidence, train the proposal calibrator on a KITTI train report and
apply it to the KITTI val report. The current quick holdout uses a 512-sample
train subset and writes:

```text
reports/perception_proposal_calibration_kitti_train512_detector_log
reports/perception_calibrated_fusion_kitti_train512_to_val1496_detector_log
reports/perception_claim_readiness_rollup
```

That run keeps the FP/precision benefit on val, but still loses recall versus
HumanISP, so it is not a broad superiority claim.

Use the claim gate to make that decision reproducible:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.claim_gate \
  reports/perception_calibrated_fusion_kitti_train512_to_val1496_features/score_label_aux \
  --profile broad_superiority \
  --target-input perception_calibrated_score_label_aux_fusion_rgb_aux \
  --baseline-input human_rgb \
  --min-samples 1000 \
  --bootstrap-samples 2000 \
  --bootstrap-seed kitti_train512_to_val1496_human \
  --require-ci \
  --output-dir reports/perception_claim_gate_kitti_train512_score_label_aux_to_val1496_vs_human
```

The `broad_superiority` profile requires the target to match or beat HumanISP on
P50, R50, R75, small-object R50, and FP/sample. It is intentionally
conservative and metric-only; passing it would still not be a product safety
claim. `--require-ci` uses paired sample-level bootstrap confidence intervals
when sample metrics are available. Add `--fail-on-fail` when using it as a
CI/readiness gate that should return a non-zero exit code on failure.

Use the narrower `fp_reducer` profile when the intended claim is not HumanISP
superiority, but recall-budgeted FP reduction versus the uncalibrated RGB+Aux
fusion path:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.claim_gate \
  reports/perception_calibrated_fusion_kitti_train512_to_val1496_features/score_label_aux \
  --profile fp_reducer \
  --target-input perception_calibrated_score_label_aux_fusion_rgb_aux \
  --baseline-input perception_fusion_rgb_aux \
  --min-samples 1000 \
  --bootstrap-samples 2000 \
  --bootstrap-seed kitti_train512_to_val1496_fusion \
  --require-ci \
  --output-dir reports/perception_claim_gate_kitti_train512_score_label_aux_to_val1496_fp_reducer_vs_fusion
```

`fp_reducer` requires non-worse precision, no more than `0.01` absolute recall
loss on R50/R75/small R50, and at least `0.10` fewer FP/sample. This supports a
bounded FP-reduction claim only; it does not convert the result into a broad
HumanISP superiority claim.

A second detector-side calibration branch trains on `perception_rgb` proposals
instead of the RGB+aux fusion proposals. It is useful because the uncalibrated
Perception RGB stream is already near HumanISP recall parity, and the calibrator
can remove false positives with a smaller recall loss:

```bash
PYTHONPATH=src python3 -m perception_isp.proposal_calibration \
  reports/perception_compare_kitti_train512_detector_log_harness \
  --input perception_rgb \
  --feature-sets score,score_label \
  --thresholds 0.00:0.30:0.01 \
  --baseline-input human_rgb \
  --train-fraction 0.67 \
  --split-strategy hash \
  --seed kitti_train512_perception_rgb_calibration \
  --recall-delta-floor -0.001 \
  --artifact-feature-set score_label \
  --artifact-output-input perception_calibrated_score_label_perception_rgb \
  --output-dir reports/perception_proposal_calibration_kitti_train512_perception_rgb
```

Applied to KITTI val 1496, this target passes the `fp_reducer` gate against
HumanISP while still failing the stricter broad-superiority gate:

```text
reports/perception_calibrated_perception_rgb_kitti_train512_to_val1496/index.html
reports/perception_claim_gate_kitti_train512_perception_rgb_fp_reducer_vs_human/index.html
reports/perception_claim_readiness_perception_rgb_fp_vs_human/index.html
```

Create a consolidated readiness dashboard from the claim gates, RGB+aux
training rollup, and calibration rollup:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.claim_dashboard \
  --claim-gate 'Human broad superiority=reports/perception_claim_gate_kitti_train512_score_label_aux_to_val1496_vs_human' \
  --claim-gate 'FP reducer vs RGB+Aux Fusion=reports/perception_claim_gate_kitti_train512_score_label_aux_to_val1496_fp_reducer_vs_fusion' \
  --training-rollup reports/perception_rgb_aux_training_rollup_kitti_val128 \
  --task-metrics reports/perception_task_metrics_kitti_train512_score_label_aux_to_val1496 \
  --comparison-rollup 'Calibration feature ablation=reports/perception_train512_calibration_feature_ablation_rollup' \
  --output-dir reports/perception_claim_readiness_dashboard
```

The dashboard currently says: broad HumanISP superiority is not supported,
bounded FP reduction is supported, and the learned RGB+aux DNN path is
trainable but not yet claim-quality. When task metrics are provided, it also
keeps the task-level claim narrow: VRU/person recall improvement versus
HumanISP is not supported when those task recall deltas are negative, even if
FP/sample is reduced.

Task-oriented group metrics can be extracted from the same saved detections:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.task_metrics \
  reports/perception_calibrated_fusion_kitti_train512_to_val1496_features/score_label_aux \
  --baseline-input human_rgb \
  --inputs human_rgb,perception_fusion_rgb_aux,perception_calibrated_score_label_aux_fusion_rgb_aux \
  --output-dir reports/perception_task_metrics_kitti_train512_score_label_aux_to_val1496
```

The current task metric result reinforces the caution: calibrated
score+label+aux reduces FP/sample, but VRU recall is still lower than HumanISP
(`dR50=-0.0138` for `vru`, `dR50=-0.0157` for `person`).

For the current KITTI evidence bundle, the one-shot readiness command rebuilds
both claim gates, task metrics, the RGB+aux training rollup, benchmark-protocol
coverage, and a dashboard that includes the task-metric tradeoff decision:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.claim_readiness \
  reports/perception_calibrated_fusion_kitti_train512_to_val1496_features/score_label_aux \
  --target-input perception_calibrated_score_label_aux_fusion_rgb_aux \
  --human-baseline-input human_rgb \
  --fusion-baseline-input perception_fusion_rgb_aux \
  --min-samples 1000 \
  --bootstrap-samples 2000 \
  --bootstrap-seed kitti_train512_to_val1496 \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log_dense_rgb_aux_ablation \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log_dense_rgb_only_ablation \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log_dense_aux_only_ablation \
  --training-summary reports/perception_rgb_aux_dense_kitti_val128_rgb_aux_ablation_eval_conf050 \
  --training-summary reports/perception_rgb_aux_dense_kitti_val128_rgb_only_ablation_eval_conf050 \
  --training-summary reports/perception_rgb_aux_dense_kitti_val128_aux_only_ablation_eval_conf050 \
  --comparison-rollup 'Calibration feature ablation=reports/perception_train512_calibration_feature_ablation_rollup' \
  --protocol-comparison-report reports/perception_compare_kitti_val1496_naive_raw_like \
  --output-dir reports/perception_claim_readiness_with_naive
```

This is a runnable SW reference, not a product ISP. The intentional next step is to compare these outputs against task metrics such as small-object recall, VRU recall, traffic-light state accuracy, and AEB early-warning lead time.

The readiness bundle now writes
`benchmark_protocol/protocol_coverage_summary.json` and
`benchmark_protocol/index.html`. This checklist encodes the minimum evidence
matrix from the RAW/perception-ISP literature: paired HumanISP and
PerceptionISP streams, enough held-out samples, fixed detector recipe,
CI-backed claim gates, task metrics, naive RAW/minimal adaptation, classical
lightweight RAW transform, and a task-aware or aux-assisted path. Missing rows
are blockers for broad HumanISP or RAW/sensor-native superiority claims.

The protocol checker can also be run directly when assembling evidence by hand:

```bash
PYTHONPATH=src python3 -m perception_isp.benchmark_protocol \
  --comparison-report reports/perception_calibrated_fusion_kitti_train512_to_val1496_features/score_label_aux \
  --comparison-report reports/perception_compare_kitti_val1496_naive_raw_like \
  --comparison-rollup 'Calibration feature ablation=reports/perception_train512_calibration_feature_ablation_rollup' \
  --training-rollup reports/perception_claim_readiness_with_naive/rgb_aux_training_rollup \
  --claim-gate reports/perception_claim_readiness_with_naive/broad_superiority_vs_human \
  --claim-gate reports/perception_claim_readiness_with_naive/fp_reducer_vs_fusion \
  --task-metrics reports/perception_claim_readiness_with_naive/task_metrics \
  --min-samples 1000 \
  --output-dir reports/perception_benchmark_protocol
```

The current 1496-image naive RAW-like baseline is:

```text
reports/perception_compare_kitti_val1496_naive_raw_like/index.html
```

It uses the same KITTI val split, YOLO11n detector, CameraE2E raw cache, and
HumanISP baseline, but sets the PerceptionISP path to linear tone mapping,
bilinear demosaic, and no denoise. It is expected to be weak; it exists to make
the RAW/perception-ISP ablation matrix honest.
