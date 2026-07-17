# PerceptionISP Pipeline Showcase — nuScenes RGB → CameraE2E RAW — Evidence Bundle

Status: **pass**<br>
Provenance class: `nuscenes_camerae2e_pipeline_showcase`<br>
Referenced source JPEGs: **6**<br>
Referenced nuScenes tables: **6**

## Reproduction command

```bash
${CAMERAE2E_ROOT}/.venv/bin/python -m perception_isp report showcase data/nuscenes --scene-name scene-0061 --camera CAM_FRONT --frames 6 --width 320 --height 180 --output-dir ${TEMP_ROOT}/perceptionisp-showcase-fixed.yLh9b0
```

## Exact configuration

```json
{
  "camera_channel": "CAM_FRONT",
  "cfa_pattern": "auto",
  "dataset_root": "${PROJECT_ROOT}/data/nuscenes",
  "dataset_version": "v1.0-mini",
  "frames": 6,
  "fresh_capture_required": true,
  "height": 180,
  "noise_seed": 0,
  "reproduction_argv": [
    "${CAMERAE2E_ROOT}/.venv/bin/python",
    "-m",
    "perception_isp",
    "report",
    "showcase",
    "data/nuscenes",
    "--scene-name",
    "scene-0061",
    "--camera",
    "CAM_FRONT",
    "--frames",
    "6",
    "--width",
    "320",
    "--height",
    "180",
    "--output-dir",
    "${TEMP_ROOT}/perceptionisp-showcase-fixed.yLh9b0"
  ],
  "scene_name": "scene-0061",
  "scene_reference_white_luminance": 10000.0,
  "scene_token": null,
  "synthetic_fallback_allowed": false,
  "width": 320
}
```

## Evidence contents

- `index.html` and `assets/`: human-readable report and visual previews.
- Lossless numerical arrays are intentionally omitted from this public preview; use the separately retained local acceptance bundle for numerical-array review.
- `source_manifest.json`: relative source/table paths, byte sizes, SHA256 values, and plane provenance.
- `environment.json`: exact argv/configuration, runtime versions, and Git state.
- `metadata_origins.csv`: field/value/origin audit.
- `exposure_plane_sources.csv`: per-plane source, exposure, seed, CFA, and RAW-domain provenance.
- `artifacts_manifest.json` and `checksums.sha256`: artifact metadata and integrity checks.

## Interpretation and licensing boundaries

- nuScenes camera inputs are processed JPEG images. They are not source-camera sensor RAW.
- CameraE2E forward-simulates `sensor.volts`; PerceptionISP maps those voltages to simulated pseudo-RAW codes. This is not native RAW recovery and cannot restore information already clipped or quantized in the JPEG.
- A dynamic HDR stress stack uses different timestamps and is unregistered. It is not a same-instant calibrated HDR bracket, radiance-recovery result, geometric-registration result, or deghosting result.
- Source JPEG and nuScenes table files are not copied into this bundle. Only dataset-relative paths, sizes, and hashes are recorded.
- Access, use, and redistribution of nuScenes data remain subject to the applicable nuScenes terms. Verify those terms before sharing this bundle or any derived visual preview.
- PNG files are visualization-only previews; this public bundle does not provide lossless numerical evidence for quantitative re-analysis.

## Public preview scope

Lossless numerical arrays from the full local acceptance bundle are not included here. See [DATA_NOTICE.md](DATA_NOTICE.md) for attribution, license, nuScenes terms, citation, and non-endorsement language for the PNG previews.

The 13 published source-RGB paths have a unique frame-image count of 6. Source-RGB input thumbnails larger than 800px on their longest edge are downscaled for publication; Aux-map, pseudo-RAW, and PerceptionISP result PNGs are retained byte-for-byte.
