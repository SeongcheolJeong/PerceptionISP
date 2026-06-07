# PerceptionISP

Software reference implementation of a perception-oriented automotive ISP.

This repo follows the design notes in `camera_sim_perception_isp_full_conversation_summary.md` and the attached Perception ISP text:

- RGB-compatible vision stream is kept for existing DNN backbones.
- Sensor-native auxiliary maps are preserved: noise, SNR, saturation, HDR source, edge, color confidence, IR/Clear, blur/focus, flicker, timing.
- Fast safety path and accurate autonomy path are separate outputs.
- CameraE2E under `/Users/seongcheoljeong/Documents/CameraE2E` can be used opportunistically, with synthetic RAW fallback.
- `docs/literature_basis.md` records the current RAW/sensor-native perception
  claim boundary: task-aware adaptation can help, but naive RAW and untrained
  aux maps are not performance claims.

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
ir_or_clear, blur_focus_confidence,
mtf_confidence, psf_sigma, psf_blur_confidence, psf_edge_likelihood
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
aux_blur_focus_confidence, aux_psf_blur_confidence,
aux_psf_edge_likelihood
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
extended 15-channel sensor-native tensor. The default remains `rgb_aux_chw` so
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
extended 15-channel sensor-native tensor.

The current MPS timing is good for iteration, not for a performance claim. From
the observed KITTI ablation runs, compact dense training is about
`59 sample-epochs/s`: a 5,985-sample, 5-epoch compact run is about 8.4 minutes
for training only, 50 epochs is about 1.4 hours, and 100 epochs is about
2.8 hours. Tensor export is separate from training and is currently about
0.33 s/sample on the cached 128-sample KITTI run, so export plus 5-epoch
training is about 41.6 minutes for 5,985 images.

Roll up export, training, and dense-eval summaries into one timing/diagnostic
report, including the extended 15-channel sensor-native tensor run:

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
  exports/perception_rgb_aux_kitti_val128_detector_log_extended \
  exports/perception_rgb_aux_kitti_val128_detector_log_dense_extended_rgb_aux \
  reports/perception_rgb_aux_dense_kitti_val128_extended_rgb_aux_eval_conf050 \
  --output-dir reports/perception_rgb_aux_training_rollup_kitti_val128_with_extended
```

The rollup is a resource and diagnostic view. It now includes a Training-Time
Plan derived from observed sample-epochs/sec and export samples/sec, making
compact KITTI-sized timing scenarios reproducible. It also makes clear that the
compact dense path trains quickly, while its direct detector metrics are still
too weak for a HumanISP-vs-PerceptionISP performance claim. The current code
exports `rgb_aux_extended_chw` as a 15-channel DNN input including PSF blur and
PSF edge-likelihood channels. Older KITTI extended artifacts were produced with
the prior 13-channel layout, so their direct dense-detector metrics remain
historical diagnostics until the export/train/eval bundle is refreshed.

A current 15-channel CameraE2E-backed smoke artifact is available here:

```text
exports/perception_rgb_aux_15ch_camerae2e_grbg_smoke_export/index.html
exports/perception_rgb_aux_15ch_camerae2e_grbg_smoke_train/train_smoke_summary.json
reports/perception_rgb_aux_15ch_camerae2e_grbg_smoke_rollup/index.html
```

It uses 4 true CFA mosaics with source/target `GRBG` and no pattern remap,
exports `rgb_aux_extended_chw` as `15 x 64 x 96`, includes
`aux_psf_blur_confidence` and `aux_psf_edge_likelihood`, and trains the tiny
smoke path for 1 epoch on MPS. This proves the current 15-channel data path;
it is not detector-performance evidence.

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
apply it to the KITTI val report. An earlier comparison branch uses a
512-sample train subset and writes:

```text
reports/perception_proposal_calibration_kitti_train512_detector_log
reports/perception_calibrated_fusion_kitti_train512_to_val1496_detector_log
reports/perception_claim_readiness_rollup
```

That pre-`t001` run keeps the FP/precision benefit on val, but still loses
recall versus HumanISP, so it is not a broad superiority claim. It is retained
as a comparison branch; the current HumanISP-relative FP-reducer branch is the
lower-threshold `t001` artifact below.

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

For an aux-including HumanISP-relative branch, use a lower threshold on the
train512 `score_label_aux` artifact. The `0.04` artifact is a strong FP reducer
against the fusion baseline, but its R50 loss is too large for the HumanISP
`fp_reducer` gate. The `0.01` threshold keeps the paired-CI recall loss inside
budget:

```bash
PYTHONPATH=src python3 -m perception_isp.proposal_calibration \
  reports/perception_compare_kitti_train512_detector_log_harness \
  --input perception_fusion_rgb_aux \
  --feature-sets score_aux,score_label,score_label_aux \
  --thresholds 0.00:0.30:0.01 \
  --baseline-input human_rgb \
  --train-fraction 0.67 \
  --split-strategy hash \
  --seed kitti_train512_calibration \
  --recall-delta-floor -0.001 \
  --artifact-feature-set score_label_aux \
  --artifact-threshold 0.01 \
  --artifact-output-input perception_calibrated_score_label_aux_fusion_rgb_aux_t001 \
  --output-dir reports/perception_proposal_calibration_kitti_train512_score_label_aux_t001
```

Applied to KITTI val 1496, this aux-assisted target passes the HumanISP
`fp_reducer` gate:

```text
reports/perception_calibrated_fusion_kitti_train512_score_label_aux_t001_to_val1496/index.html
reports/perception_claim_gate_kitti_train512_score_label_aux_t001_fp_reducer_vs_human/index.html
reports/perception_claim_readiness_score_label_aux_t001_fp_vs_human/index.html
```

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
training rollup, mechanism validation, CFA/edge diagnostics, benchmark
protocol, and calibration rollup:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.claim_dashboard \
  --claim-gate 'Aux broad vs Human=reports/perception_claim_gate_kitti_train512_score_label_aux_t001_broad_vs_human' \
  --claim-gate 'Aux FP reducer vs Human=reports/perception_claim_gate_kitti_train512_score_label_aux_t001_fp_reducer_vs_human' \
  --training-rollup reports/perception_rgb_aux_training_rollup_kitti_val128_with_extended \
  --task-metrics reports/perception_task_metrics_kitti_train512_score_label_aux_t001_vs_human \
  --task-gate reports/perception_task_gate_kitti_train512_score_label_aux_t001_recall_vs_human \
  --mechanism-validation reports/perception_mechanism_validation_synthetic \
  --cfa-stress-sweep reports/perception_cfa_stress_sweep_synthetic \
  --edge-confidence-suite reports/perception_edge_confidence_suite_synthetic \
  --edge-fidelity-suite reports/perception_edge_fidelity_suite_synthetic \
  --scene-edge-confidence reports/perception_scene_edge_confidence_bus_highinfo \
  --scene-edge-confidence reports/perception_scene_edge_confidence_bus_cfa_psf_sweep \
  --scene-information-stress reports/perception_scene_information_stress_synthetic \
  --aux-contribution-audit reports/perception_aux_contribution_audit_kitti_train512_to_val1496 \
  --protocol-coverage reports/perception_benchmark_protocol_kitti_with_naive_extended \
  --comparison-rollup 'Calibration feature ablation=reports/perception_train512_calibration_feature_ablation_rollup' \
  --output-dir reports/perception_claim_readiness_score_label_aux_t001_fp_vs_human_extended
```

The dashboard currently says: broad HumanISP superiority is not supported,
bounded FP reduction versus HumanISP is supported, and the learned RGB+aux DNN path is
trainable but not yet claim-quality. When task metrics are provided, it also
keeps the task-level claim narrow: the `recall_improvement` task gate fails for
VRU/person/cyclist/vehicle/small-object groups, so task recall improvement
versus HumanISP is not supported even though FP/sample is reduced. Mechanism
validation is shown as front-end feasibility evidence only, not as detector
performance evidence, and the CFA stress sweep is shown as diagnostic
condition/CFA evidence only. The edge-confidence suite is also shown as
diagnostic difficult-edge evidence, the object edge-fidelity suite is shown as
diagnostic CFA/LensPSF edge evidence, the scene-information stress suite is
shown as diagnostic scene-to-sensor evidence, and the aux contribution audit is
shown as detector-side calibration evidence, not as DNN detector-performance
evidence. When scene-edge deltas and aux contribution deltas are both positive
in the expected direction, the dashboard also reports a diagnostic
front-end/downstream bridge; this is co-observed evidence, not same-sample
causal correlation. The dashboard also includes a `Performance Evidence Map`
section that summarizes the recommended claim posture, blocked claims, current
evidence rows, and the next evidence to build. For the current bundle, the
recommended posture is a narrow recall-budgeted FP-reduction claim with
front-end/aux feasibility support; broad HumanISP superiority remains blocked.

Task-oriented group metrics can be extracted from the same saved detections:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.task_metrics \
  reports/perception_calibrated_fusion_kitti_train512_score_label_aux_t001_to_val1496 \
  --baseline-input human_rgb \
  --inputs human_rgb,perception_fusion_rgb_aux,perception_calibrated_score_label_aux_fusion_rgb_aux_t001 \
  --output-dir reports/perception_task_metrics_kitti_train512_score_label_aux_t001_vs_human
```

The current task metric result reinforces the caution: calibrated
score+label+aux reduces FP/sample, but VRU recall is still lower than HumanISP
(`dR50=-0.0061` for `vru`, `dR50=-0.0063` for `person`).

Before spending on large DNN fine-tuning, use the mechanism-validation suite to
show why PerceptionISP signals are feasible and useful. This suite creates
paired synthetic sensor stressors and checks whether the corresponding
aux/confidence maps respond in the expected direction:

```bash
PYTHONPATH=src python3 -m perception_isp.mechanism_validation \
  --width 160 \
  --height 96 \
  --cfa RGGB \
  --cfa GRBG \
  --cfa RCCB \
  --cfa RGBIR \
  --cfa MONO \
  --output-dir reports/perception_mechanism_validation_synthetic
```

The current synthetic mechanism report is:

```text
reports/perception_mechanism_validation_synthetic/index.html
```

It passes four front-end mechanism checks: low-light SNR/visibility response,
glare saturation/clipping response, low-MTF edge/demosaic/focus-confidence
response, and configured CFA decode/map validity. This is feasibility evidence
for the PerceptionISP front-end signals, not a detector-performance claim.

To inspect CFA-dependent front-end behavior rather than only pass/fail support,
run the synthetic CFA stress sweep:

```bash
PYTHONPATH=src python3 -m perception_isp.cfa_stress_sweep \
  --width 160 \
  --height 96 \
  --cfa RGGB \
  --cfa GRBG \
  --cfa RCCB \
  --cfa RGBIR \
  --cfa MONO \
  --condition nominal_hdr \
  --condition low_light \
  --condition glare \
  --condition low_mtf \
  --output-dir reports/perception_cfa_stress_sweep_synthetic
```

The current CFA stress report is:

```text
reports/perception_cfa_stress_sweep_synthetic/index.html
```

It ranks front-end signal quality by condition. In the current synthetic scene,
`MONO` ranks highest for low-light and low-MTF structure scores, while `RGBIR`
ranks highest for the glare score. Treat this as CFA-dependent signal
feasibility evidence, not as a detector result or a product sensor
recommendation.

To validate the difficult-edge confidence use case directly, run the synthetic
edge-confidence suite:

```bash
PYTHONPATH=src python3 -m perception_isp.edge_confidence_suite \
  --width 160 \
  --height 96 \
  --cfa RGGB \
  --output-dir reports/perception_edge_confidence_suite_synthetic
```

The current edge-confidence report is:

```text
reports/perception_edge_confidence_suite_synthetic/index.html
```

It checks that low light, glare saturation, and low-MTF blur lower the
PerceptionISP edge/confidence evidence in the expected direction. Current
synthetic deltas are directional: low light lowers mean edge confidence by
`-0.1531`, glare lowers mean edge confidence by `-0.1381`, and low MTF lowers
strong-edge confidence by `-0.4119`. This is front-end confidence evidence for
hard edge cases, not detector-performance evidence.

To compare object-boundary edge fidelity across HumanISP, PerceptionISP, aux
edge maps, CFA pattern, and LensPSF blur, run the object edge-fidelity suite:

```bash
PYTHONPATH=src python3 -m perception_isp.edge_fidelity_suite \
  --sensor-width 160 \
  --sensor-height 96 \
  --oversample 6 \
  --cfa RGGB \
  --cfa GRBG \
  --cfa BGGR \
  --cfa GBRG \
  --cfa RCCB \
  --cfa RGBIR \
  --cfa MONO \
  --psf-sigma 0.0 \
  --psf-sigma 0.8 \
  --psf-sigma 1.6 \
  --output-dir reports/perception_edge_fidelity_suite_synthetic
```

The current object edge-fidelity report is:

```text
reports/perception_edge_fidelity_suite_synthetic/index.html
```

It builds a labeled object-edge oracle and a sensor-edge oracle, then measures
HumanISP RGB edge F1, PerceptionISP RGB edge F1, and aux edge-map F1 against
those oracles. In the current synthetic case, LensPSF sigma from `0.0` to
`1.6` sensor pixels reduces absolute sensor-edge P95 from `0.0546` to `0.0362`
(`ratio=0.6624`), so the PSF effect is visible. If PSF sigma is much smaller
than the sensor pixel pitch, or if PSF is applied only after low-resolution
sensor sampling, this effect can be nearly invisible. The report's LensPSF
Visibility table makes this explicit with ratio-vs-nominal and delta-vs-previous
edge-contrast metrics. The PerceptionISP edge block also consumes calibration
`psf_sigma_map` as a PSF blur-confidence prior and exports `psf_blur_confidence`
and `psf_edge_likelihood` aux maps, including the extended RGB+aux DNN tensor.
Treat this as front-end edge-fidelity evidence across CFA/LensPSF, not
detector-performance evidence.

To compare edge evidence on a higher-information real scene, run the scene-edge
confidence suite:

```bash
PYTHONPATH=src:/Users/seongcheoljeong/Documents/CameraE2E/src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.scene_edge_confidence_suite \
  --source sample-image \
  --image-path data/sample_images/bus.jpg \
  --width 320 \
  --height 240 \
  --scene-scale 2 \
  --cfa auto \
  --psf-sigma 0 \
  --psf-sigma 0.8 \
  --psf-sigma 1.6 \
  --tone-mapping detector_log \
  --denoise-strength 0.30 \
  --demosaic-method edge_aware \
  --demosaic-artifact-suppression 0.20 \
  --output-dir reports/perception_scene_edge_confidence_bus_highinfo
```

The current report is:

```text
reports/perception_scene_edge_confidence_bus_highinfo/index.html
reports/perception_scene_edge_confidence_bus_cfa_psf_sweep/index.html
```

The CameraE2E report feeds a `640 x 480` scene image through CameraE2E and
evaluates a `320 x 240` sensor output, with source/target CFA both `GRBG` and
no pattern remap. It now sweeps LensPSF sigma `0.0`, `0.8`, and `1.6` sensor
pixels. The high-resolution scene edge map is downsampled as the proxy oracle.
Across those PSF conditions, HumanISP RGB edge-proxy F1 is `0.6519`,
PerceptionISP RGB edge-proxy F1 is `0.6650`, PerceptionISP aux edge-strength F1
is `0.7473`, and PerceptionISP aux edge-confidence F1 is `0.3727`. The
PerceptionISP RGB minus HumanISP RGB source-edge F1 delta is `+0.0131`, the aux
edge-strength delta is `+0.0954`, and both win rates are `1.0000` across the
three LensPSF cases. The LensPSF confidence-response check also passes, so
increasing PSF reduces PerceptionISP edge-confidence as expected.

The separate `bus_cfa_psf_sweep` report intentionally uses direct RGB-to-RAW
remosaicing rather than CameraE2E true-CFA output so CFA patterns can be swept
without source/target CFA remapping. It covers `BGGR`, `GBRG`, `GRBG`, and
`RGGB` at the same PSF sigmas. Across the 12 CFA/PSF cases, PerceptionISP RGB
source-edge F1 delta is `+0.0122`, aux edge-strength delta is `+0.2049`, and
both win rates are `1.0000`. Treat both reports as front-end scene-edge
evidence, not object-boundary ground truth or detector-performance evidence.

To verify that the test is not merely passing an RGB scene through both ISPs,
run the scene-information stress suite. It creates a higher-resolution scene
oracle, samples it through a lower-resolution CFA sensor model, then runs
PerceptionISP on the resulting RAW:

```bash
PYTHONPATH=src python3 -m perception_isp.scene_information_stress \
  --sensor-width 160 \
  --sensor-height 96 \
  --oversample 8 \
  --cfa RGGB \
  --output-dir reports/perception_scene_information_stress_synthetic
```

The current scene-information report is:

```text
reports/perception_scene_information_stress_synthetic/index.html
```

It passes three diagnostic checks. In `supersampled_thin_detail`, scene luma
gradient P90 is `0.44` while the sensor luma gradient P90 is `0.0`, showing
detail above sensor sampling being integrated away. In `cfa_chroma_alias`,
scene chroma gradient P90 is `0.86` while PerceptionISP color confidence is
`0.0`, showing CFA/color uncertainty rather than recoverable object evidence.
In `subpixel_signal`, signal contrast retention is `0.0937`, showing
fill-factor loss for a sub-pixel bright signal. This is scene-to-sensor
diagnostic evidence, not detector-performance evidence, and it explicitly
shows that no ISP can recover information absent from RAW.

To verify that aux features are actually used in a downstream scoring path,
audit the proposal-calibration feature ablation:

```bash
PYTHONPATH=src python3 -m perception_isp.aux_contribution_audit \
  reports/perception_train512_calibration_feature_ablation_rollup \
  --calibration-summary reports/perception_proposal_calibration_kitti_train512_score_label_aux_t001 \
  --recall-floor -0.005 \
  --min-fp-reduction 0.02 \
  --output-dir reports/perception_aux_contribution_audit_kitti_train512_to_val1496
```

The current aux contribution report is:

```text
reports/perception_aux_contribution_audit_kitti_train512_to_val1496/index.html
```

It passes the downstream calibration checks. On the KITTI train512-to-val1496
feature ablation, `score_aux` versus uncalibrated RGB+Aux fusion gives
`dP=+0.0035`, `dR50=-0.0027`, and `dFP=-0.0608`. Adding aux to
`score_label` gives `dP=+0.0054`, `dR50=-0.0022`, and `dFP=-0.0622` versus
`score_label` alone. This proves aux is being used by the proposal scoring
branch. Combined with the scene-edge reports, the dashboard now shows a
directionally positive bridge: scene-edge RGB F1 delta `+0.0124`, aux
edge-strength delta `+0.1830`, and incremental aux proposal-scoring dFP@0.50
`-0.0622`.
The same-sample bridge inside the aux contribution audit compares
`score_label` and `score_label_aux` proposal outputs on the same 1496 KITTI
samples: adding aux removes 111 proposals, 95 of them false positives and 16 of
them true positives, for net `fp_delta_count=-93` and `tp_delta_count=-16`.
So the incremental aux removal set is 85.6% false positives. Those removed false
positives also have lower edge support than kept true positives
(`removed_fp_minus_kept_tp_edge_support_mean=-0.0596`), and low edge support
identifies removed FP proposals versus kept TP proposals with AUC `0.6904`
(`0.7701` versus kept FP proposals).
This still does not prove a trained RGB+aux DNN detector claim or same-sample
causality, but it is same-sample proposal-level evidence that aux edge support
is connected to the FP proposals being removed.

Make that task-level decision reproducible with the task gate:

```bash
PYTHONPATH=src python3 -m perception_isp.task_gate \
  reports/perception_task_metrics_kitti_train512_score_label_aux_t001_vs_human \
  --profile recall_improvement \
  --target-input perception_calibrated_score_label_aux_fusion_rgb_aux_t001 \
  --baseline-input human_rgb \
  --min-group-gt 1 \
  --output-dir reports/perception_task_gate_kitti_train512_score_label_aux_t001_recall_vs_human
```

The gate currently fails the `recall_improvement` profile for `vru`,
`person`, `cyclist`, `vehicle`, and `small_all`; `traffic_light` is skipped
because there is no positive GT in this KITTI slice.

For the current KITTI evidence bundle, the one-shot readiness command rebuilds
both claim gates, task metrics, the task gate, condition metrics, the condition
robustness gate, the RGB+aux training rollup, benchmark-protocol coverage, and
a dashboard that includes the task-metric tradeoff decision:

```bash
PYTHONPATH=src \
/Users/seongcheoljeong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m perception_isp.claim_readiness \
  reports/perception_calibrated_fusion_kitti_train512_score_label_aux_t001_to_val1496 \
  --target-input perception_calibrated_score_label_aux_fusion_rgb_aux_t001 \
  --human-baseline-input human_rgb \
  --fusion-baseline-input perception_fusion_rgb_aux \
  --min-samples 1000 \
  --bootstrap-samples 2000 \
  --bootstrap-seed kitti_train512_to_val1496_t001 \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log_dense_rgb_aux_ablation \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log_dense_rgb_only_ablation \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log_dense_aux_only_ablation \
  --training-summary reports/perception_rgb_aux_dense_kitti_val128_rgb_aux_ablation_eval_conf050 \
  --training-summary reports/perception_rgb_aux_dense_kitti_val128_rgb_only_ablation_eval_conf050 \
  --training-summary reports/perception_rgb_aux_dense_kitti_val128_aux_only_ablation_eval_conf050 \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log_extended \
  --training-summary exports/perception_rgb_aux_kitti_val128_detector_log_dense_extended_rgb_aux \
  --training-summary reports/perception_rgb_aux_dense_kitti_val128_extended_rgb_aux_eval_conf050 \
  --comparison-rollup 'Calibration feature ablation=reports/perception_train512_calibration_feature_ablation_rollup' \
  --protocol-comparison-report reports/perception_compare_kitti_val1496_naive_raw_like \
  --mechanism-validation reports/perception_mechanism_validation_synthetic \
  --cfa-stress-sweep reports/perception_cfa_stress_sweep_synthetic \
  --edge-confidence-suite reports/perception_edge_confidence_suite_synthetic \
  --edge-fidelity-suite reports/perception_edge_fidelity_suite_synthetic \
  --scene-edge-confidence reports/perception_scene_edge_confidence_bus_highinfo \
  --scene-edge-confidence reports/perception_scene_edge_confidence_bus_cfa_psf_sweep \
  --scene-information-stress reports/perception_scene_information_stress_synthetic \
  --aux-contribution-audit reports/perception_aux_contribution_audit_kitti_train512_to_val1496 \
  --output-dir reports/perception_claim_readiness_with_naive_extended
```

This is a runnable SW reference, not a product ISP. The intentional next step is to compare these outputs against task metrics such as small-object recall, VRU recall, traffic-light state accuracy, and AEB early-warning lead time.

The readiness bundle now writes
`benchmark_protocol/protocol_coverage_summary.json` and
`benchmark_protocol/index.html`. This checklist encodes the minimum evidence
matrix from the RAW/perception-ISP literature: paired HumanISP and
PerceptionISP streams, enough held-out samples, fixed detector recipe,
CI-backed claim gates, task metrics, task gate, condition-specific metrics, a
condition robustness gate, front-end mechanism validation, naive RAW/minimal
adaptation, classical lightweight RAW transform, and a task-aware or
aux-assisted path. Missing rows are blockers for broad HumanISP or
RAW/sensor-native superiority claims. Recommended diagnostic rows, such as the
CFA stress sweep, edge-confidence suite, object edge-fidelity suite,
scene edge-confidence suite, scene-information stress suite, and aux contribution audit, help interpret
sensor-native signals but do not create a detector-performance claim.

The protocol checker can also be run directly when assembling evidence by hand:

```bash
PYTHONPATH=src python3 -m perception_isp.benchmark_protocol \
  --comparison-report reports/perception_calibrated_fusion_kitti_train512_score_label_aux_t001_to_val1496 \
  --comparison-report reports/perception_compare_kitti_val1496_naive_raw_like \
  --comparison-rollup 'Calibration feature ablation=reports/perception_train512_calibration_feature_ablation_rollup' \
  --training-rollup reports/perception_rgb_aux_training_rollup_kitti_val128_with_extended \
  --claim-gate reports/perception_claim_gate_kitti_train512_score_label_aux_t001_broad_vs_human \
  --claim-gate reports/perception_claim_gate_kitti_train512_score_label_aux_t001_fp_reducer_vs_human \
  --task-metrics reports/perception_task_metrics_kitti_train512_score_label_aux_t001_vs_human \
  --task-gate reports/perception_task_gate_kitti_train512_score_label_aux_t001_recall_vs_human \
  --condition-metrics reports/perception_condition_metrics_kitti_train512_score_label_aux_t001_vs_human \
  --condition-gate reports/perception_condition_gate_kitti_train512_score_label_aux_t001_fp_reducer_vs_human \
  --mechanism-validation reports/perception_mechanism_validation_synthetic \
  --cfa-stress-sweep reports/perception_cfa_stress_sweep_synthetic \
  --edge-confidence-suite reports/perception_edge_confidence_suite_synthetic \
  --edge-fidelity-suite reports/perception_edge_fidelity_suite_synthetic \
  --scene-edge-confidence reports/perception_scene_edge_confidence_bus_highinfo \
  --scene-edge-confidence reports/perception_scene_edge_confidence_bus_cfa_psf_sweep \
  --scene-information-stress reports/perception_scene_information_stress_synthetic \
  --aux-contribution-audit reports/perception_aux_contribution_audit_kitti_train512_to_val1496 \
  --min-samples 1000 \
  --output-dir reports/perception_benchmark_protocol_kitti_with_naive_extended
```

The latest extended-inclusive protocol report is:

```text
reports/perception_benchmark_protocol_kitti_with_naive_extended/index.html
reports/perception_claim_readiness_score_label_aux_t001_fp_vs_human_extended/index.html
reports/perception_task_gate_kitti_train512_score_label_aux_t001_recall_vs_human/index.html
reports/perception_condition_metrics_kitti_train512_score_label_aux_t001_vs_human/index.html
reports/perception_condition_gate_kitti_train512_score_label_aux_t001_fp_reducer_vs_human/index.html
reports/perception_mechanism_validation_synthetic/index.html
reports/perception_cfa_stress_sweep_synthetic/index.html
reports/perception_edge_confidence_suite_synthetic/index.html
reports/perception_edge_fidelity_suite_synthetic/index.html
reports/perception_scene_edge_confidence_bus_highinfo/index.html
reports/perception_scene_information_stress_synthetic/index.html
reports/perception_aux_contribution_audit_kitti_train512_to_val1496/index.html
```

It marks `coverage_status=coverage_complete`, including front-end mechanism
validation, the recommended extended sensor-native tensor row, CFA stress
sweep, edge-confidence suite, object edge-fidelity suite, scene edge-confidence
suite, scene-information stress suite, and aux contribution audit, while
`metric_claim_status=fp_reducer_only`.
That is an evidence-coverage result, not a broad-superiority result; the
dashboard still says broad HumanISP superiority is not supported, while
recall-budgeted FP reduction versus HumanISP is supported. It also reports a
diagnostic front-end/downstream bridge between positive scene-edge deltas and
downstream FP reduction, plus a same-sample aux proposal bridge where
incremental aux scoring removes 95 FP and 16 TP proposals and those removed FP
have lower edge support than kept TP with low-edge AUC `0.6904`, without
treating that as a trained-DNN or broad-superiority proof. The condition gate passes the
`fp_reducer` profile on 8 evaluated condition slices; the
`warning:over_exposure` slice is skipped because it has only 7 samples.
The same dashboard's `Performance Evidence Map` lists 12 current evidence rows
and five next evidence targets: scene-edge oracle to proposal correlation,
CFA/LensPSF detector sweep, RGB+Aux DNN fine-tune gate, high-information
real-scene expansion, and a failure/slice casebook. The previous aux-edge
same-sample proposal correlation is now part of the current evidence row.

The current 1496-image naive RAW-like baseline is:

```text
reports/perception_compare_kitti_val1496_naive_raw_like/index.html
```

It uses the same KITTI val split, YOLO11n detector, CameraE2E raw cache, and
HumanISP baseline, but sets the PerceptionISP path to linear tone mapping,
bilinear demosaic, and no denoise. It is expected to be weak; it exists to make
the RAW/perception-ISP ablation matrix honest.
